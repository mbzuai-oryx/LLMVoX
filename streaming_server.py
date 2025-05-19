"""
streaming-server.py-
Streaming TTS Server Application

This is a FastAPI application that provides a streaming text-to-speech API
endpoint. It uses a combination of a streaming LLM for text generation and
two TTS models that alternate to provide low-latency speech synthesis.
"""
import time
import asyncio
import threading
from queue import Queue, Empty
import argparse
import torch
import torch.nn.functional as F
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from inference.model_handler import ModelHandler
from inference.llm_streaming import StreamModel
from configs.inference_config import config
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from inference.asr import ASR
import re
def parse_arguments():
    """Parse command line arguments and override config."""
    parser = argparse.ArgumentParser(description="Streaming TTS API Server")
    
    # Model paths
    parser.add_argument("--chat_type", type=str, help="Specify input modalities for chat from ['text','voice','multimodal','visual_speech']")
    parser.add_argument("--wav_config_path", type=str, help="Path to wave tokenizer config")
    parser.add_argument("--wav_model_path", type=str, help="Path to wave tokenizer model")
    parser.add_argument("--encoder_model_path", type=str, help="Path to encoder model")
    parser.add_argument("--tokenizer_path", type=str, help="Path to tokenizer")
    parser.add_argument("--llmvox_checkpoint_path", type=str, help="Path to GPT checkpoint")

    # LLM settings
    parser.add_argument("--llm_checkpoint", type=str, help="LLM checkpoint path")
    parser.add_argument("--llm_device", type=str, help="Device for LLM (e.g., cuda:0)")
    parser.add_argument("--llm_max_tokens", type=int, help="Maximum tokens for LLM generation")
    parser.add_argument("--llm_temperature", type=float, help="Temperature for LLM sampling")
    parser.add_argument("--llm_top_p", type=float, help="Top-p for LLM sampling")
    parser.add_argument("--llm_top_k", type=int, help="Top-k for LLM sampling")

    # TTS model settings
    parser.add_argument("--tts_device_1", type=int, help="GPU ID for TTS model 1")
    parser.add_argument("--tts_device_2", type=int, help="GPU ID for TTS model 2")

    # Streaming settings
    parser.add_argument("--system_prompt", type=str, help="System prompt for LLM")
    parser.add_argument("--initial_dump_size_1", type=int, help="Initial chunk size for model 1")
    parser.add_argument("--initial_dump_size_2", type=int, help="Initial chunk size for model 2")
    parser.add_argument("--max_dump_size", type=int, help="Maximum chunk size")
    parser.add_argument("--max_audio_length", type=int, help="Maximum audio length")

    # Special tokens
    parser.add_argument("--eos_token", type=str, help="End of sequence token")
    parser.add_argument("--pad_token_id", type=int, help="Padding token ID")
    parser.add_argument("--eoa_token_id", type=int, help="End of audio token ID")

    # API settings
    parser.add_argument("--api_host", type=str, help="API host address")
    parser.add_argument("--api_port", type=int, help="API port number")

    # ASR settings
    parser.add_argument("--asr_model", type=str, default="small", help="Whisper model variant: tiny, base, small, medium, large")
    parser.add_argument("--asr_device", type=str, default="cuda:2", help="Device for ASR model")
    parser.add_argument("--asr_sample_rate", type=float, default=16000.0, help="Audio sample rate in Hz")
    parser.add_argument("--asr_max_audio_length", type=int, default=60, help="Maximum audio length in seconds")
    parser.add_argument("--asr_default_language", type=str, default="english", help="Default language for transcription")
    parser.add_argument("--asr_enable_translation", type=bool, default=False, help="Enable translation instead of transcription by default")
    
    args = parser.parse_args()
    
    # Update config with command line arguments, only for non-None values
    # This preserves default values from inference_config.py when not specified
    global config
    args_dict = vars(args)
    for key, value in args_dict.items():
        if value is not None:
            config[key] = value
    
    return config


# Initialize FastAPI app
app = FastAPI(
    title="Streaming TTS API",
    description="API for streaming text-to-speech synthesis with real-time LLM generation",
    version="1.0.0",
)

# Add CORS middleware to allow requests from web browsers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def clean_text(text, eos_token=config["eos_token"]):
    """
    Clean and normalize text from the LLM for better TTS performance.
    
    Args:
        text: Text to clean
        eos_token: End of sequence token
        
    Returns:
        Cleaned text optimized for TTS
    """
    # Basic cleaning
    text = text.strip()
    text = text.replace("**", "")
    # text = text.replace("!", ".")
    text = text.replace("-", " ")
    
    # Remove periods after numbers (e.g., "5." becomes "5")
    text = re.sub(r'(\d)\.(?=\s|$)', r'\1', text)
    
    # Handle special characters that may cause issues
    text = re.sub(r'\*', '', text)                 # Remove asterisks
    text = re.sub(r'#', ' number ', text)          # Replace hashtags
    text = re.sub(r'&', ' and ', text)             # Replace ampersands
    text = re.sub(r'@', ' at ', text)              # Replace @ symbols
    
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Handle ellipses (prevent awkward pauses)
    text = re.sub(r'\.{3,}', ' pause ', text)
    
    # Handle other TTS-unfriendly patterns
    text = re.sub(r'(\d),(\d)', r'\1\2', text)     # Remove commas in numbers
    text = re.sub(r'\/+', ' slash ', text)         # Replace slashes
    text = re.sub(r'\\+', ' backslash ', text)     # Replace backslashes
    
    # Remove remaining non-alphanumeric characters except allowed punctuation
    # Uncomment and modify as needed
    # allowed_chars = ".,?;:()\"' "
    # pattern = f"[^A-Za-z0-9{re.escape(allowed_chars)}]"
    # text = re.sub(pattern, "", text)
    
    return text

# Global model handlers (initialized on startup)
model_handler_1 = None
model_handler_2 = None
stream_llm_model = None

@app.on_event("startup")
async def startup_event():
    """Initialize models when the server starts up."""
    global model_handler_1, model_handler_2, stream_llm_model,asr_model
    
    # Parse device IDs from config
    device_id_1 = config["tts_device_1"]
    device_id_2 = config["tts_device_2"]
    
    print("Loading models...")
    
    # Initialize model handlers on specified devices
    model_handler_1 = ModelHandler(config,device_id=device_id_1)
    model_handler_2 = ModelHandler(config,device_id=device_id_2)
    
    # Initialize LLM
    if config["chat_type"] in ['voice','text']:
        stream_llm_model = model_handler_2.initialize_stream_model()
        if config['chat_type'] == 'voice':
            asr_model=ASR(config)
    elif config['chat_type'] == 'multimodal':
        stream_llm_model = model_handler_2.initialize_stream_multimodal()
    elif config['chat_type'] == 'visual_speech':
        asr_model=ASR(config)
        stream_llm_model = model_handler_2.initialize_vlm_model()
    
    print("All models loaded successfully!")

def text_streamer_producer(
    request, 
    stream_model: StreamModel, 
    text_token_queue_1: Queue, 
    text_token_queue_2: Queue
):
    """
    Thread function to produce text tokens from the LLM and route them to TTS models.
    
    Args:
        text_input: User's text prompt
        stream_model: Streaming LLM model
        text_token_queue_1: Queue for the first TTS model
        text_token_queue_2: Queue for the second TTS model
    """
    
    # List of token queues to alternate between
    active_queue = [text_token_queue_1, text_token_queue_2]  # Start with the first queue
    active_index = 0
    
    # Collect all text outputs for logging
    text_outputs = []
    system_text = config["system_prompt"]
    if config['chat_type'] in ['voice','text']:
        # Get system prompt from config
        if "text" in request:
            prompt_text = request.text
            print(f"Received TTS request: {prompt_text}")
        else:
            prompt_text=asr_model.run_asr(request)
        text_streamer = stream_model.predict({"system": system_text, "prompt": prompt_text})
    elif config['chat_type'] == 'multimodal':
        audio_data = request.audio_base64
        image_list = request.image_list
        text_streamer = stream_model.predict({"system": system_text,"audio_data":audio_data,"images_data":image_list})
    elif config['chat_type'] == 'visual_speech':
        prompt_text=asr_model.run_asr(request)
        image_data = request.image_base64
        text_streamer = stream_model.predict({"system": system_text,"prompt":prompt_text,"image_base64":image_data})

    # Process streaming tokens
    eos = config["eos_token"]
    with torch.inference_mode():
        for output in text_streamer:
            # Skip empty outputs
            if output in ['', '-']:
                continue

            output=output.strip()
            # Clean the output text
            if output!=eos:
                output = clean_text(output, eos)
            # Store for logging
            text_outputs.append(output)
            # Add to queue if not empty
            if output:
                # Put text token into active queue
                active_queue[active_index].put(output)
                # Switch to alternate queue when sentence ends
                if output.endswith('.'):
                    active_index = 1 if active_index == 0 else 0
    
    # Log the complete output
    print("Text generation complete.")
    print("Generated text: " + " ".join(text_outputs))

def audio_generator_sync(
    index: int,
    dump_size: int,
    model_handler: ModelHandler, 
    text_token_queue: Queue, 
    audio_byte_queue: Queue
):
    """
    Thread function to generate audio from text tokens.
    
    Args:
        index: Model index (0 or 1)
        dump_size: Initial size of audio chunks to dump
        model_handler: Model handler for TTS generation
        text_token_queue: Queue for receiving text tokens
        audio_byte_queue: Queue for sending audio byte chunks
    """
    # Initialize state variables
    speech_gen_index = 0
    current_speech_token = None
    token_batch = []
    bandwidth_id = torch.tensor([0]).to(model_handler.device)
    kvcache = None
    end_of_speech = False
    speech_outputs = []
    max_audio_len = config["max_audio_length"]
    complete_sentence = []
    active_model = True
    text_output = []
    end_generation = False
    eos = config["eos_token"]
    pad_token_id = config["pad_token_id"]
    eoa_token_id = config["eoa_token_id"]
    
    start = time.time()
    
    with torch.inference_mode():
        while True:
            if active_model:
                # Retrieve a text token from the queue
                if not end_of_speech:
                    text_token = text_token_queue.get()
                    text_output.append(text_token)
                
                if not end_of_speech:
                    # Check if end of sentence or EOS token
                    if (eos in text_token) or (text_token[-1] == "."):
                        if eos in text_token:
                            end_generation = True
                        text_token = text_token.rstrip(eos)
                        end_of_speech = True
                    else:
                        end_of_speech = False
                    
                    # Convert text token to phonemes
                    text_phone = text_token.strip()
                    text_phone_tokens = model_handler.tokenizer(text_phone)["input_ids"]
                    
                    # Add EOS token if end of speech
                    if end_of_speech:
                        text_phone_tokens = text_phone_tokens + [385]
                    
                    # Prepare tensor for model input
                    text_phone_tokens = torch.tensor(text_phone_tokens)
                    text_tokens = torch.tensor(text_phone_tokens).unsqueeze(0).to(model_handler.device)
                    text_embeddings = model_handler.llm_model(text_tokens)
                else:
                    # Use padding token for end of speech
                    pad_token = torch.tensor([pad_token_id]).unsqueeze(0).to(model_handler.device)
                    pad_token_embeddings = model_handler.llm_model(pad_token)
                    text_embeddings = pad_token_embeddings
                
                # Process each embedding token
                for i in range(text_embeddings.shape[1]):
                    # Initial speech embedding or use previous token
                    if speech_gen_index == 0:
                        speech_embed = torch.zeros((1, 1, 512), device=model_handler.device)
                    else:
                        speech_token = torch.tensor([[current_speech_token]]).to(model_handler.device)
                        speech_embed = model_handler.wavtokenizer.codes_to_features(speech_token).permute(0, 2, 1).to(model_handler.device)
                    
                    # Combine text and speech embeddings
                    text_embed = text_embeddings[:, i, :].unsqueeze(1)
                    speech_decoder_input = torch.cat([text_embed, speech_embed], dim=2)
                    speech_decoder_input = F.normalize(speech_decoder_input, p=2, dim=2, eps=1e-8)
                    
                    # Add previous context
                    if speech_gen_index > 0:
                        speech_decoder_input = torch.cat([speech_decoder_input_prev, speech_decoder_input], dim=1)
                    
                    # Generate next speech token
                    speech_decoder_output, _, kvcache = model_handler.model(speech_decoder_input, kvcache=kvcache)
                    logits = speech_decoder_output[:, -1, :]
                    probs = F.softmax(logits, dim=-1)
                    
                    # Get most probable token
                    current_speech_token = probs.argmax(dim=-1).item()
                    speech_outputs.append(current_speech_token)
                    
                    if index == 0:
                        complete_sentence.append(current_speech_token)
                    
                    # Store current input for next iteration
                    speech_decoder_input_prev = speech_decoder_input
                    speech_gen_index += 1
                    
                    # Check if we need to dump audio (enough tokens accumulated)
                    if len(speech_outputs) >= dump_size:
                        print(f"Received audio at {time.time()-start:.2f}s")
                        token_batch = speech_outputs[:dump_size]
                        speech_outputs = speech_outputs[dump_size:]
                        
                        # Convert tokens to audio
                        predicted_tokens = torch.tensor([token_batch]).to(model_handler.device)
                        features = model_handler.wavtokenizer.codes_to_features(predicted_tokens)
                        audio_out = model_handler.wavtokenizer.decode(features, bandwidth_id=bandwidth_id).squeeze(0)
                        
                        # Convert to bytes and enqueue
                        audio_bytes = audio_out.cpu().numpy().astype('float32').tobytes()
                        audio_byte_queue.put(audio_bytes)
                        print(f"Dumped audio at {time.time()-start:.2f}s")
                        
                        # Increase dump size for faster streaming (up to max)
                        max_dump = config["max_dump_size"]
                        if dump_size < max_dump:
                            dump_size = min(dump_size * 3, max_dump)
                            print(f"Increased chunk size to: {dump_size}")
                    
                    # Check for end-of-audio token
                    elif eoa_token_id in speech_outputs:
                        token_batch = speech_outputs
                        predicted_tokens = torch.tensor([token_batch]).to(model_handler.device)
                        features = model_handler.wavtokenizer.codes_to_features(predicted_tokens)
                        audio_out = model_handler.wavtokenizer.decode(features, bandwidth_id=bandwidth_id).squeeze(0)
                        speech_outputs = []
                        
                        # Convert to bytes and enqueue
                        audio = audio_out.cpu().numpy().astype('float32')
                        audio_bytes = audio.tobytes()
                        audio_byte_queue.put(audio_bytes)
                        
                        # Increase dump size
                        max_dump = config["max_dump_size"]
                        if dump_size < max_dump:
                            dump_size = min(dump_size * 3, max_dump)
                    
                    # Check for termination conditions
                    if current_speech_token == eoa_token_id or len(speech_outputs) > max_audio_len:
                        if end_generation:
                            print("End of generation detected")
                            audio_byte_queue.put("end")
                        elif index == 0:
                            audio_byte_queue.put(1)  # Signal to switch to model 1
                        elif index == 1:
                            audio_byte_queue.put(0)  # Signal to switch to model 0
                        
                        # Reset state for next generation
                        active_model = True
                        speech_gen_index = 0
                        current_speech_token = None
                        token_batch = []
                        bandwidth_id = torch.tensor([0]).to(model_handler.device)
                        kvcache = None
                        end_of_speech = False
                        speech_outputs = []
                        end_generation = False
                        complete_sentence = []
                        text_output = []
                        
                        # Adjust dump size for next segment
                        max_dump = config["max_dump_size"]
                        if dump_size < max_dump:
                            dump_size = min(dump_size * 3, max_dump)
    
    # Signal end of stream
    audio_byte_queue.put(None)
    print("Audio generator finished.")

async def audio_generator_async(audio_byte_queue_1, audio_byte_queue_2):
    """
    Asynchronous generator that yields audio bytes from the queues.
    
    Args:
        audio_byte_queue_1: Queue for the first TTS model
        audio_byte_queue_2: Queue for the second TTS model
        
    Yields:
        Audio data bytes or control signals
    """
    queue_list = [audio_byte_queue_1, audio_byte_queue_2]
    current_queue = queue_list[0]
    loop = asyncio.get_event_loop()
    
    while True:
        try:
            # Retrieve data from the queue with a timeout
            audio_bytes = await loop.run_in_executor(None, current_queue.get, True, 1)
            
            if audio_bytes == 'end':
                print("End of generation")
                break
                
            if audio_bytes in [0, 1]:
                # Switch active queue
                current_queue = queue_list[audio_bytes]
                continue
                
            if audio_bytes is None:
                continue
                
            yield audio_bytes
            
        except Empty:
            continue
        except Exception as e:
            print(f"Error in audio_generator_async: {e}")
            break
    
    print("Audio generator async finished.")

class TTSRequest(BaseModel):
    """Request model for TTS endpoint."""
    text: str

# Define a model for the request body
class STTRequest(BaseModel):
    source_language: str
    target_language: str
    audio_base64: str


class MMRequest(BaseModel):
    audio_base64: str
    image_list:list


class VLMSRequest(BaseModel):
    source_language: str
    target_language: str
    audio_base64: str
    image_base64: str


@app.post("/tts")
async def tts(request: TTSRequest):
    """
    Generate speech from text with streaming output.
    
    Args:
        request: TTSRequest with text field
        
    Returns:
        StreamingResponse with audio data
    """
    # Create queues for inter-thread communication
    text_token_q_1 = Queue()
    audio_byte_q_1 = Queue()
    
    text_token_q_2 = Queue()
    audio_byte_q_2 = Queue()
    
    # Start the text producer thread
    producer_thread = threading.Thread(
        target=text_streamer_producer,
        args=(request, stream_llm_model, text_token_q_1, text_token_q_2),
        daemon=True
    )
    producer_thread.start()
    
    # Start the audio generator threads
    consumer_thread_1 = threading.Thread(
        target=audio_generator_sync,
        args=(0, config["initial_dump_size_1"], model_handler_1, text_token_q_1, audio_byte_q_1),
        daemon=True
    )
    
    consumer_thread_2 = threading.Thread(
        target=audio_generator_sync,
        args=(1, config["initial_dump_size_2"], model_handler_2, text_token_q_2, audio_byte_q_2),
        daemon=True
    )
    
    consumer_thread_1.start()
    consumer_thread_2.start()
    
    # Return streaming response
    return StreamingResponse(
        audio_generator_async(audio_byte_q_1, audio_byte_q_2),
        media_type="application/octet-stream",
    )


@app.post("/voicechat")
async def stt_api(request: STTRequest):

    # Create queues for inter-thread communication
    text_token_q_1 = Queue()
    audio_byte_q_1 = Queue()
    
    text_token_q_2 = Queue()
    audio_byte_q_2 = Queue()
    

    # Start the text producer thread
    producer_thread = threading.Thread(
        target=text_streamer_producer,
        args=(request, stream_llm_model, text_token_q_1, text_token_q_2),
        daemon=True
    )
    producer_thread.start()
    
    # Start the audio generator threads
    consumer_thread_1 = threading.Thread(
        target=audio_generator_sync,
        args=(0, config["initial_dump_size_1"], model_handler_1, text_token_q_1, audio_byte_q_1),
        daemon=True
    )
    
    consumer_thread_2 = threading.Thread(
        target=audio_generator_sync,
        args=(1, config["initial_dump_size_2"], model_handler_2, text_token_q_2, audio_byte_q_2),
        daemon=True
    )
    
    consumer_thread_1.start()
    consumer_thread_2.start()
    
    # Return streaming response
    return StreamingResponse(
        audio_generator_async(audio_byte_q_1, audio_byte_q_2),
        media_type="application/octet-stream",
    )

@app.post("/multimodalchat")
async def stt_api(request: MMRequest):

    # Create queues for inter-thread communication
    text_token_q_1 = Queue()
    audio_byte_q_1 = Queue()
    
    text_token_q_2 = Queue()
    audio_byte_q_2 = Queue()


    producer_thread = threading.Thread(
        target=text_streamer_producer,
        args=(request, stream_llm_model, text_token_q_1, text_token_q_2),
        daemon=True
    )
    producer_thread.start()
    
    # Start the audio generator threads
    consumer_thread_1 = threading.Thread(
        target=audio_generator_sync,
        args=(0, config["initial_dump_size_1"], model_handler_1, text_token_q_1, audio_byte_q_1),
        daemon=True
    )
    
    consumer_thread_2 = threading.Thread(
        target=audio_generator_sync,
        args=(1, config["initial_dump_size_2"], model_handler_2, text_token_q_2, audio_byte_q_2),
        daemon=True
    )
    
    consumer_thread_1.start()
    consumer_thread_2.start()
    
    # Return streaming response
    return StreamingResponse(
        audio_generator_async(audio_byte_q_1, audio_byte_q_2),
        media_type="application/octet-stream",
    )

@app.post("/vlmschat")
async def stt_api(request: VLMSRequest):

    # Create queues for inter-thread communication
    text_token_q_1 = Queue()
    audio_byte_q_1 = Queue()
    
    text_token_q_2 = Queue()
    audio_byte_q_2 = Queue()


    producer_thread = threading.Thread(
        target=text_streamer_producer,
        args=(request, stream_llm_model, text_token_q_1, text_token_q_2),
        daemon=True
    )
    producer_thread.start()
    
    # Start the audio generator threads
    consumer_thread_1 = threading.Thread(
        target=audio_generator_sync,
        args=(0, config["initial_dump_size_1"], model_handler_1, text_token_q_1, audio_byte_q_1),
        daemon=True
    )
    
    consumer_thread_2 = threading.Thread(
        target=audio_generator_sync,
        args=(1, config["initial_dump_size_2"], model_handler_2, text_token_q_2, audio_byte_q_2),
        daemon=True
    )
    
    consumer_thread_1.start()
    consumer_thread_2.start()
    
    # Return streaming response
    return StreamingResponse(
        audio_generator_async(audio_byte_q_1, audio_byte_q_2),
        media_type="application/octet-stream",
    )


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Streaming TTS API",
        "usage": "POST /tts with {\"text\": \"Your question or prompt here\"}",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    
    # Parse command line arguments and update config
    config = parse_arguments()
    
    # Print the active configuration for debugging
    print("Active configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Start the server
    uvicorn.run(
        app, 
        host=config["api_host"], 
        port=config["api_port"]
    )

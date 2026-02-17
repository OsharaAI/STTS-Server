
import requests
import numpy as np
import time

def test_streaming(text="Hello, this is a test of the streaming TTS system in STTS server."):
    url = "http://localhost:4000/tts/stream"
    print(f"Testing streaming TTS at {url}...")
    
    payload = {
        "text": text,
        "chunk_size": 20,
    }
    
    start_time = time.time()
    first_byte_time = None
    total_bytes = 0
    
    try:
        with requests.post(url, json=payload, stream=True) as response:
            if response.status_code != 200:
                print(f"Error: {response.status_code} - {response.text}")
                return
            
            print("Response received, reading stream...")
            
            for chunk in response.iter_content(chunk_size=None):
                if not chunk:
                    continue
                
                if first_byte_time is None:
                    first_byte_time = time.time()
                    latency = first_byte_time - start_time
                    print(f"Time to first chunk: {latency*1000:.2f} ms")
                
                total_bytes += len(chunk)
                # print(f"Received chunk: {len(chunk)} bytes")
                
    except Exception as e:
        print(f"Request failed: {e}")
        return

    total_time = time.time() - start_time
    print(f"Streaming complete.")
    print(f"Total time: {total_time:.2f}s")
    print(f"Total bytes: {total_bytes}")
    
    if total_bytes > 0:
        # Assuming float32
        samples = total_bytes / 4
        duration = samples / 24000
        print(f"Estimated audio duration: {duration:.2f}s")
        print(f"RTF: {total_time / duration:.3f}")
    else:
        print("No audio received!")

if __name__ == "__main__":
    test_streaming()

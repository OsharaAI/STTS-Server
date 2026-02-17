#!/usr/bin/env python3
"""
Test client for PCM TTS API
"""

import requests
import numpy as np
import wave
import argparse


def test_pcm_tts(
    server_url: str = "http://localhost:5000",
    text: str = "Hello, this is a test of the PCM audio API.",
    sample_rate: int = 24000,
    output_file: str = "test_output.wav"
):
    """
    Test the PCM TTS API endpoint.
    
    Args:
        server_url: Base URL of the STTS server
        text: Text to synthesize
        sample_rate: Target sample rate
        output_file: Output WAV file path
    """
    
    print(f"Testing PCM TTS API at {server_url}")
    print(f"Text: '{text}'")
    print(f"Sample rate: {sample_rate} Hz")
    print("-" * 60)
    
    # Test info endpoint first
    try:
        print("\n1. Getting API info...")
        info_response = requests.get(f"{server_url}/pcm-tts/info")
        info_response.raise_for_status()
        info = info_response.json()
        print(f"   ✓ API endpoint available")
        print(f"   Format: {info['format']['encoding']}, {info['format']['bit_depth']}-bit")
        print(f"   Default sample rate: {info['format']['default_sample_rate']} Hz")
    except Exception as e:
        print(f"   ✗ Error getting API info: {e}")
        return False
    
    # Generate PCM audio
    try:
        print("\n2. Generating PCM audio...")
        payload = {
            "text": text,
            "sample_rate": sample_rate
        }
        
        response = requests.post(
            f"{server_url}/pcm-tts/generate",
            json=payload
        )
        response.raise_for_status()
        
        # Get metadata from headers
        actual_sample_rate = int(response.headers.get('X-Sample-Rate', sample_rate))
        channels = int(response.headers.get('X-Channels', 1))
        bit_depth = int(response.headers.get('X-Bit-Depth', 16))
        duration = float(response.headers.get('X-Duration', 0))
        
        print(f"   ✓ Audio generated successfully")
        print(f"   Sample rate: {actual_sample_rate} Hz")
        print(f"   Channels: {channels}")
        print(f"   Bit depth: {bit_depth}")
        print(f"   Duration: {duration:.2f} seconds")
        print(f"   Data size: {len(response.content)} bytes")
        
        # Convert PCM bytes to numpy array
        pcm_data = np.frombuffer(response.content, dtype=np.int16)
        print(f"   Samples: {len(pcm_data)}")
        
        # Save as WAV file for testing
        print(f"\n3. Saving to {output_file}...")
        with wave.open(output_file, 'wb') as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(bit_depth // 8)  # Convert bits to bytes
            wav_file.setframerate(actual_sample_rate)
            wav_file.writeframes(response.content)
        
        print(f"   ✓ Saved successfully")
        
        # Verify audio data
        print("\n4. Verifying audio data...")
        audio_float = pcm_data.astype(np.float32) / 32768.0
        peak_amplitude = np.max(np.abs(audio_float))
        rms_level = np.sqrt(np.mean(audio_float ** 2))
        
        print(f"   Peak amplitude: {peak_amplitude:.3f}")
        print(f"   RMS level: {rms_level:.3f}")
        print(f"   Min value: {np.min(pcm_data)}")
        print(f"   Max value: {np.max(pcm_data)}")
        
        if peak_amplitude > 0.01 and peak_amplitude <= 1.0:
            print(f"   ✓ Audio levels look good")
        else:
            print(f"   ⚠ Warning: Unusual audio levels detected")
        
        print("\n" + "=" * 60)
        print("✓ PCM TTS API test completed successfully!")
        print(f"✓ Audio saved to: {output_file}")
        print(f"✓ You can play it with: ffplay {output_file}")
        print("=" * 60)
        
        return True
        
    except requests.exceptions.HTTPError as e:
        print(f"   ✗ HTTP Error: {e}")
        print(f"   Response: {e.response.text}")
        return False
    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Test PCM TTS API")
    parser.add_argument(
        "--server",
        default="http://localhost:5000",
        help="Server URL (default: http://localhost:5000)"
    )
    parser.add_argument(
        "--text",
        default="Hello, this is a test of the PCM audio API.",
        help="Text to synthesize"
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=24000,
        help="Target sample rate in Hz (default: 24000)"
    )
    parser.add_argument(
        "--output",
        default="test_output.wav",
        help="Output WAV file path (default: test_output.wav)"
    )
    
    args = parser.parse_args()
    
    success = test_pcm_tts(
        server_url=args.server,
        text=args.text,
        sample_rate=args.sample_rate,
        output_file=args.output
    )
    
    exit(0 if success else 1)


if __name__ == "__main__":
    main()

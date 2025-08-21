#!/usr/bin/env python3
import os
import sys
import signal
import time
import tempfile
from pathlib import Path
from typing import Optional, Literal
from enum import Enum
import threading

import typer
import pyperclip
from dotenv import load_dotenv

from .audio import AudioRecorder
from .adapters import FasterWhisperAdapter, FasterWhisperGPUAdapter, OpenAIAdapter, STTAdapter

load_dotenv()

app = typer.Typer(help="paDC - Speech to Text CLI")

PID_FILE = Path("/tmp/padc.pid")
AUDIO_FILE = Path("/tmp/padc_recording.wav")


class AdapterType(str, Enum):
    local = "local"
    local_gpu = "local_gpu"
    openai = "openai"


def get_default_adapter() -> AdapterType:
    """Get the default adapter from environment or return 'local'"""
    env_adapter = os.environ.get("PADC_ADAPTER", "local").lower()
    if env_adapter in ["local", "local_gpu", "openai"]:
        return AdapterType(env_adapter)
    return AdapterType.local


def get_adapter(adapter_type: AdapterType) -> STTAdapter:
    if adapter_type == AdapterType.local:
        model_size = os.environ.get("WHISPER_MODEL", "base")
        return FasterWhisperAdapter(model_size=model_size)
    elif adapter_type == AdapterType.local_gpu:
        model_size = os.environ.get("WHISPER_MODEL", "base")
        return FasterWhisperGPUAdapter(model_size=model_size)
    elif adapter_type == AdapterType.openai:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            typer.echo("Error: OPENAI_API_KEY environment variable not set", err=True)
            raise typer.Exit(1)
        return OpenAIAdapter(api_key=api_key)
    else:
        raise ValueError(f"Unknown adapter type: {adapter_type}")


def transcribe_and_copy(audio_path: Path, adapter: STTAdapter, paste: bool = False, insert: bool = False):
    try:
        typer.echo("Transcribing audio...")
        text = adapter.transcribe(audio_path)
        
        if text:
            pyperclip.copy(text)
            typer.echo(f"Transcription: {text}")
            typer.echo("✓ Copied to clipboard")
            
            if paste:
                import subprocess
                try:
                    subprocess.run(["xdotool", "type", text], check=True)
                    typer.echo("✓ Typed with xdotool")
                except subprocess.CalledProcessError:
                    typer.echo("⚠ Failed to type with xdotool (is xdotool installed?)", err=True)
                except FileNotFoundError:
                    typer.echo("⚠ xdotool not found. Install it to use --paste option", err=True)
            
            if insert:
                import subprocess
                try:
                    subprocess.run(["xdotool", "key", "shift+Insert"], check=True)
                    typer.echo("✓ Pasted with Shift+Insert")
                except subprocess.CalledProcessError:
                    typer.echo("⚠ Failed to paste with xdotool (is xdotool installed?)", err=True)
                except FileNotFoundError:
                    typer.echo("⚠ xdotool not found. Install it to use --insert option", err=True)
        else:
            typer.echo("No speech detected")
    except Exception as e:
        typer.echo(f"Error during transcription: {e}", err=True)


@app.command()
def start(
    adapter: AdapterType = typer.Option(None, "--adapter", "-a", help="STT adapter to use")
):
    """Start recording in daemon mode"""
    if adapter is None:
        adapter = get_default_adapter()
    
    typer.echo(f"Using adapter: {adapter.value}")
    
    if PID_FILE.exists():
        typer.echo("Daemon already running. Use 'stop' command first.", err=True)
        raise typer.Exit(1)
    
    pid = os.fork()
    if pid > 0:
        PID_FILE.write_text(str(pid))
        typer.echo(f"Started recording daemon (PID: {pid})")
        typer.echo("Use 'padc stop' to stop recording and transcribe")
        sys.exit(0)
    
    os.setsid()
    
    recorder = AudioRecorder()
    recorder.start()
    
    def signal_handler(signum, frame):
        audio_data = recorder.stop()
        if recorder.save_to_wav(audio_data, AUDIO_FILE):
            with open("/tmp/padc_adapter.txt", "w") as f:
                f.write(adapter.value)
        sys.exit(0)
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    while True:
        time.sleep(1)


@app.command()
def stop(
    paste: bool = typer.Option(False, "--paste", "-p", help="Also type the text using xdotool"),
    insert: bool = typer.Option(False, "--insert", "-i", help="Paste using Shift+Insert")
):
    """Stop recording and transcribe"""
    if not PID_FILE.exists():
        typer.echo("No daemon running", err=True)
        raise typer.Exit(1)
    
    try:
        pid = int(PID_FILE.read_text())
        os.kill(pid, signal.SIGTERM)
        typer.echo("Stopping recording...")
        
        time.sleep(0.5)

        recorder = AudioRecorder()
        recorder.start()
        
        if AUDIO_FILE.exists():
            adapter_type = AdapterType.local
            adapter_file = Path("/tmp/padc_adapter.txt")
            if adapter_file.exists():
                adapter_type = AdapterType(adapter_file.read_text().strip())
                adapter_file.unlink()
            
            typer.echo(f"Using adapter: {adapter_type.value}")
            adapter = get_adapter(adapter_type)
            transcribe_and_copy(AUDIO_FILE, adapter, paste=paste, insert=insert)
            AUDIO_FILE.unlink()
        
        PID_FILE.unlink()

        recorder.stop()
        
    except ProcessLookupError:
        typer.echo("Daemon process not found", err=True)
        PID_FILE.unlink()
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"Error stopping daemon: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def toggle(
    adapter: AdapterType = typer.Option(None, "--adapter", "-a", help="STT adapter to use"),
    paste: bool = typer.Option(False, "--paste", "-p", help="Also type the text using xdotool (when stopping)"),
    insert: bool = typer.Option(False, "--insert", "-i", help="Paste using Shift+Insert (when stopping)")
):
    """Toggle daemon recording - starts if not running, stops if running"""
    if adapter is None:
        adapter = get_default_adapter()
    
    if PID_FILE.exists():
        # Daemon is running, stop it
        stop(paste=paste, insert=insert)
    else:
        # Daemon is not running, start it
        start(adapter=adapter)


@app.command()
def live(
    adapter: AdapterType = typer.Option(None, "--adapter", "-a", help="STT adapter to use")
):
    """Live recording mode - press Enter to start/stop"""
    if adapter is None:
        adapter = get_default_adapter()
    
    typer.echo(f"Using adapter: {adapter.value}")
    typer.echo("Live mode - Press Enter to start/stop recording, Ctrl+C to exit")
    
    stt_adapter = get_adapter(adapter)
    recorder = AudioRecorder()
    recording = False
    
    try:
        while True:
            if not recording:
                input("Press Enter to start recording...")
                typer.echo("🔴 Recording... (Press Enter to stop)")
                recorder.start()
                recording = True
            else:
                input()
                typer.echo("⏹ Stopping...")
                audio_data = recorder.stop()
                recording = False
                
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                    tmp_path = Path(tmp_file.name)
                    if recorder.save_to_wav(audio_data, tmp_path):
                        transcribe_and_copy(tmp_path, stt_adapter)
                    tmp_path.unlink()
                
                typer.echo("")
    
    except KeyboardInterrupt:
        if recording:
            recorder.stop()
        typer.echo("\n\nExiting live mode")


@app.command()
def record(
    duration: float = typer.Option(None, "--duration", "-d", help="Recording duration in seconds"),
    adapter: AdapterType = typer.Option(None, "--adapter", "-a", help="STT adapter to use")
):
    """Single recording session"""
    if adapter is None:
        adapter = get_default_adapter()
    
    typer.echo(f"Using adapter: {adapter.value}")
    
    stt_adapter = get_adapter(adapter)
    recorder = AudioRecorder()
    
    try:
        if duration:
            typer.echo(f"🔴 Recording for {duration} seconds...")
            audio_data = recorder.record_for_duration(duration)
        else:
            typer.echo("🔴 Recording... (Press Ctrl+C to stop)")
            recorder.start()
            try:
                while True:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                typer.echo("\n⏹ Stopping...")
                audio_data = recorder.stop()
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)
            if recorder.save_to_wav(audio_data, tmp_path):
                transcribe_and_copy(tmp_path, stt_adapter)
            tmp_path.unlink()
    
    except Exception as e:
        typer.echo(f"Error during recording: {e}", err=True)
        raise typer.Exit(1)


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    """Default action when no command is specified"""
    if ctx.invoked_subcommand is None:
        record(duration=None, adapter=None)


if __name__ == "__main__":
    app()

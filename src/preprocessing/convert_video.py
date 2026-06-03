import argparse
from pathlib import Path
import subprocess

def convert_avi_to_mp4(input_path: Path, output_path: Path):
    """
    Konvertiert eine .avi Datei zu .mp4 mittels ffmpeg.
    """
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input {input_path} nicht gefunden.")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        "ffmpeg",
        "-y",  # Überschreibt die Ausgabe ohne Nachfrage
        "-i", str(input_path),
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-c:a", "aac",
        str(output_path),
    ]
    
    print("Running ffmpeg...")
    subprocess.run(cmd, check=True)
    
    print(f"Fertig: {output_path}")
    
def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--input", type=str, required=True, help="Pfad zur .avi Datei")
    parser.add_argument("--output", type=str, help="Pfad zur .mp4 Ausgabe")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_suffix(".mp4")
        
    convert_avi_to_mp4(input_path, output_path)

if __name__ == "__main__":
    main()
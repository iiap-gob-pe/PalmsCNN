import subprocess
import sys
import os
import importlib
import tarfile
import urllib.request

def install_package(package_name):
    """Install a Python package using pip in user space."""
    try:
        importlib.import_module(package_name)
        print(f"{package_name} is already installed.")
    except ImportError:
        print(f"Installing {package_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name, "--user"])

def download_file(file_id, output_name):
    """Download a file from Google Drive using gdown."""
    import gdown
    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"Downloading {output_name}...")
    gdown.download(url, output_name, quiet=False)

def check_extraction_tool():
    """Check for installed extraction tools (unrar, 7z, rar)."""
    tools = [("unrar", "--version"), ("7z", "--help"), ("rar", "--version")]
    for tool, flag in tools:
        try:
            result = subprocess.run([tool, flag], capture_output=True, text=True, check=True)
            print(f"Found {tool}: {result.stdout.splitlines()[0]}")
            return tool
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
    return None

def download_and_setup_unrar():
    """Download and compile portable unrar in the working directory."""
    unrar_url = "https://www.rarlab.com/rar/unrarsrc-7.0.9.tar.gz"
    unrar_tar = "unrarsrc-7.0.9.tar.gz"
    unrar_dir = "unrar"
    unrar_path = os.path.join(unrar_dir, "unrar")

    if os.path.exists(unrar_path):
        print("Portable unrar is already set up.")
        return unrar_path

    print("Checking for gcc and make...")
    try:
        subprocess.run(["gcc", "--version"], capture_output=True, check=True)
        subprocess.run(["make", "--version"], capture_output=True, check=True)
    except Exception as e:
        print(f"Error: gcc or make not found: {e}")
        print("Cannot compile unrar. Please ensure gcc and make are installed, or use a precompiled unrar/7z/rar binary.")
        sys.exit(1)

    print("Downloading unrar source...")
    try:
        urllib.request.urlretrieve(unrar_url, unrar_tar)
    except Exception as e:
        print(f"Failed to download unrar: {e}")
        print("Please check the URL at https://www.rarlab.com/download.htm and update unrar_url in the script.")
        sys.exit(1)

    print("Extracting unrar source...")
    try:
        with tarfile.open(unrar_tar, "r:gz") as tar:
            tar.extractall()
        os.remove(unrar_tar)  # Clean up the tar file
    except Exception as e:
        print(f"Failed to extract unrar source: {e}")
        sys.exit(1)

    print("Compiling unrar...")
    try:
        os.chdir(unrar_dir)
        subprocess.check_call(["make", "-f", "makefile"])
        os.chmod("unrar", 0o755)  # Make unrar executable
        os.chdir("..")
    except Exception as e:
        print(f"Failed to compile unrar: {e}")
        sys.exit(1)

    if not os.path.exists(unrar_path):
        print("unrar executable not found after compilation.")
        sys.exit(1)

    print("Verifying unrar...")
    try:
        result = subprocess.run([unrar_path, "--version"], capture_output=True, text=True)
        print(result.stdout)
    except Exception as e:
        print(f"unrar verification failed: {e}")
        sys.exit(1)

    return unrar_path

def extract_rar(file_path, program):
    """Extract a RAR file using patoolib with the specified program."""
    import patoolib
    print(f"Extracting {file_path}...")
    try:
        patoolib.extract_archive(file_path, outdir=".", program=program)
    except Exception as e:
        print(f"Failed to extract {file_path}: {e}")
        sys.exit(1)

def main():
    # List of files to download: (Google Drive file ID, output filename)
    files_to_download = [
        ("1CAYRc3tex1gdyPG8l4pLzq6qgGl1fvT8", "data.rar"),
        ("1ReYvQhbODcFx0sTkV0d0p4vtFC_0cejy", "model_quant.rar"),
        ("1sN1JrDDs-GcQsHbcLzSjaqAnuMOni31Y", "model_segment.rar")
    ]

    # Install required Python packages
    install_package("gdown")
    install_package("patoolib")

    # Check for existing extraction tools
    extraction_tool = check_extraction_tool()
    if extraction_tool:
        program = extraction_tool
    else:
        # Download and compile unrar if no tool is found
        program = download_and_setup_unrar()

    # Download files
    for file_id, output_name in files_to_download:
        download_file(file_id, output_name)

    # Extract RAR files
    for _, output_name in files_to_download:
        if os.path.exists(output_name):
            extract_rar(output_name, program)
        else:
            print(f"Error: {output_name} not found for extraction.")
            sys.exit(1)

    # Remove RAR files
    for _, output_name in files_to_download:
        if os.path.exists(output_name):
            print(f"Removing {output_name}...")
            os.remove(output_name)
        else:
            print(f"Error: {output_name} not found for deletion.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)
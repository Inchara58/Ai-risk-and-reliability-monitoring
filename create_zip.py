"""
Helper script to create a standalone, distribution-ready ZIP archive of AEGIS AI
"""
import zipfile
import os

EXCLUDE_DIRS = {'.git', '__pycache__', '.pytest_cache', 'node_modules', '.agents', '.gemini'}
EXCLUDE_EXTS = {'.pyc', '.pyo', '.pyd', '.zip', '.log'}

def make_zip(output_filename="aegis_ai_dual_platform.zip"):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    zip_path = os.path.join(base_dir, output_filename)
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(base_dir):
            # Exclude unwanted directories
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and not d.startswith('.')]
            
            for file in files:
                if any(file.endswith(ext) for ext in EXCLUDE_EXTS):
                    continue
                if file == output_filename:
                    continue
                    
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, base_dir)
                zipf.write(full_path, rel_path)
                
    size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"Created {zip_path} ({size_mb:.2f} MB)")

if __name__ == "__main__":
    make_zip()

import sys
import importlib.util

def check_library(lib_name):
    spec = importlib.util.find_spec(lib_name)
    if spec is None:
        print(f"❌ {lib_name} is NOT installed")
        return False
    else:
        print(f"✅ {lib_name} is installed")
        return True

def main():
    print("Running Environment Checks...\n")
    print(f"Python Version: {sys.version.split(' ')[0]}\n")
    
    libraries = [
        "llama_index", 
        "qdrant_client", 
        "transformers", 
        "torch", 
        "spacy", 
        "ragas", 
        "fastapi", 
        "mlflow",
        "uvicorn",
        "dotenv"
    ]
    
    all_passed = True
    for lib in libraries:
        if not check_library(lib):
            all_passed = False
            
    print("\n---")
    if all_passed:
        print("🎉 All core libraries are successfully installed!")
    else:
        print("⚠️ Some libraries are missing. Please check the installation.")

if __name__ == "__main__":
    main()

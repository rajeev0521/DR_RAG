import os

def load_text_file(filepath):
    """Loads a text or markdown file and returns its content."""
    if not os.path.exists(filepath):
        print(f"Error: File {filepath} not found.")
        return None
        
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    return content

def main():
    print("--- Document Loading Test ---")
    
    # Paths to the sample datasets
    legal_doc_path = os.path.join('data', 'raw', 'sample_legal_contract.md')
    academic_doc_path = os.path.join('data', 'raw', 'sample_academic_paper.txt')
    
    print("\n1. Testing Legal Document Load:")
    legal_content = load_text_file(legal_doc_path)
    if legal_content:
        print(f"✅ Successfully loaded legal document.")
        print(f"   Length: {len(legal_content)} characters.")
        print(f"   Preview: {legal_content[:100]}...")
        
    print("\n2. Testing Academic Document Load:")
    academic_content = load_text_file(academic_doc_path)
    if academic_content:
        print(f"✅ Successfully loaded academic document.")
        print(f"   Length: {len(academic_content)} characters.")
        print(f"   Preview: {academic_content[:100]}...")

if __name__ == "__main__":
    main()

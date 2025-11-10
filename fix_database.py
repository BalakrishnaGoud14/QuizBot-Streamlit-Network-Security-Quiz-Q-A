
import sys
import shutil
from pathlib import Path

print("QuizBot Database Troubleshooting Tool")
print("=" * 60)

# Check if chroma directory exists
chroma_path = Path("nsrag/chroma")

if not chroma_path.exists():
    print(" ChromaDB directory not found!")
    print(f"   Expected location: {chroma_path.absolute()}")
    print("\n Solution: Run the database population script")
    print("   cd nsrag && python3 populate_database.py")
    sys.exit(1)

print(f"ChromaDB directory found: {chroma_path.absolute()}")

# Check directory contents
files = list(chroma_path.glob("*"))
print(f"\n Directory contents ({len(files)} items):")
for f in files:
    if f.is_file():
        size = f.stat().st_size / 1024  # KB
        print(f"    {f.name} ({size:.1f} KB)")
    else:
        print(f"    {f.name}/")

# Check for sqlite database
sqlite_file = chroma_path / "chroma.sqlite3"
if sqlite_file.exists():
    size_mb = sqlite_file.stat().st_size / (1024 * 1024)
    print(f"\n SQLite database found ({size_mb:.2f} MB)")
else:
    print("\n SQLite database not found")

# Offer to rebuild
print("\n" + "=" * 60)
print("Options:")
print("1. Keep existing database (recommended)")
print("2. Rebuild database from scratch (will delete existing data)")
print("3. Exit")

choice = input("\nEnter your choice (1-3): ").strip()

if choice == "2":
    confirm = input("  This will DELETE all existing data. Type 'yes' to confirm: ").strip().lower()
    if confirm == "yes":
        print("\n  Removing old database...")
        shutil.rmtree(chroma_path)
        print(" Database removed")
        print("\n Rebuilding database...")
        print("   Run: cd nsrag && python3 populate_database.py")
    else:
        print(" Cancelled")
elif choice == "1":
    print("\n Keeping existing database")
    print("\n If you still have issues, try:")
    print("   1. Restart Ollama")
    print("   2. Check Python version compatibility")
    print("   3. Update ChromaDB: pip3 install --upgrade chromadb")
else:
    print("\n Exiting")

print("\n" + "=" * 60)
print(" QuizBot Database Tool - Complete")

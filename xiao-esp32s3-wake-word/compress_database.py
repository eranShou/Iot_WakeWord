#!/usr/bin/env python3
"""
Database Compression/Decompression Script
Compresses and decompresses the database folder with zero data loss.
Supports multiple compression algorithms for optimal results.
"""

import os
import sys
import shutil
import hashlib
import zipfile
import tarfile
import gzip
import bz2
import lzma
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime

class DatabaseCompressor:
    def __init__(self, database_path: str, output_path: str = None):
        self.database_path = Path(database_path)
        self.output_path = Path(output_path) if output_path else self.database_path.parent / "compressed_database"
        self.checksums_file = self.output_path / "checksums.json"
        
        if not self.database_path.exists():
            raise FileNotFoundError(f"Database path does not exist: {self.database_path}")
    
    def calculate_file_checksum(self, file_path: Path) -> str:
        """Calculate SHA-256 checksum of a file for integrity verification."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def calculate_checksums(self) -> Dict[str, str]:
        """Calculate checksums for all files in the database."""
        print("Calculating file checksums for integrity verification...")
        checksums = {}
        
        for root, dirs, files in os.walk(self.database_path):
            for file in files:
                file_path = Path(root) / file
                relative_path = file_path.relative_to(self.database_path)
                checksums[str(relative_path)] = self.calculate_file_checksum(file_path)
        
        return checksums
    
    def save_checksums(self, checksums: Dict[str, str]):
        """Save checksums to a JSON file."""
        checksum_data = {
            "timestamp": datetime.now().isoformat(),
            "database_path": str(self.database_path),
            "total_files": len(checksums),
            "checksums": checksums
        }
        
        with open(self.checksums_file, 'w') as f:
            json.dump(checksum_data, f, indent=2)
    
    def load_checksums(self) -> Dict[str, str]:
        """Load checksums from JSON file."""
        if not self.checksums_file.exists():
            raise FileNotFoundError("Checksums file not found. Run compression first.")
        
        with open(self.checksums_file, 'r') as f:
            data = json.load(f)
        return data["checksums"]
    
    def verify_integrity(self, extracted_path: Path) -> bool:
        """Verify that all files match their original checksums."""
        print("Verifying file integrity...")
        try:
            original_checksums = self.load_checksums()
            all_match = True
            
            for relative_path, expected_checksum in original_checksums.items():
                file_path = extracted_path / relative_path
                if not file_path.exists():
                    print(f"❌ Missing file: {relative_path}")
                    all_match = False
                    continue
                
                actual_checksum = self.calculate_file_checksum(file_path)
                if actual_checksum != expected_checksum:
                    print(f"❌ Checksum mismatch: {relative_path}")
                    all_match = False
                else:
                    print(f"✅ Verified: {relative_path}")
            
            return all_match
        except Exception as e:
            print(f"❌ Integrity verification failed: {e}")
            return False
    
    def compress_zip(self, compression_level: int = 6) -> str:
        """Compress database using ZIP format."""
        print(f"Compressing database using ZIP (level {compression_level})...")
        archive_path = self.output_path / "database.zip"
        
        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=compression_level) as zipf:
            for root, dirs, files in os.walk(self.database_path):
                for file in files:
                    file_path = Path(root) / file
                    relative_path = file_path.relative_to(self.database_path)
                    zipf.write(file_path, relative_path)
        
        return str(archive_path)
    
    def decompress_zip(self, archive_path: str, extract_to: Path) -> bool:
        """Decompress ZIP archive."""
        print(f"Decompressing {archive_path}...")
        try:
            with zipfile.ZipFile(archive_path, 'r') as zipf:
                zipf.extractall(extract_to)
            return True
        except Exception as e:
            print(f"❌ Decompression failed: {e}")
            return False
    
    def compress_tar_gz(self, compression_level: int = 6) -> str:
        """Compress database using TAR.GZ format."""
        print(f"Compressing database using TAR.GZ (level {compression_level})...")
        archive_path = self.output_path / "database.tar.gz"
        
        with tarfile.open(archive_path, 'w:gz', compresslevel=compression_level) as tar:
            tar.add(self.database_path, arcname=self.database_path.name)
        
        return str(archive_path)
    
    def decompress_tar_gz(self, archive_path: str, extract_to: Path) -> bool:
        """Decompress TAR.GZ archive."""
        print(f"Decompressing {archive_path}...")
        try:
            with tarfile.open(archive_path, 'r:gz') as tar:
                tar.extractall(extract_to)
            return True
        except Exception as e:
            print(f"❌ Decompression failed: {e}")
            return False
    
    def compress_tar_xz(self) -> str:
        """Compress database using TAR.XZ format (best compression)."""
        print("Compressing database using TAR.XZ (maximum compression)...")
        archive_path = self.output_path / "database.tar.xz"
        
        with tarfile.open(archive_path, 'w:xz') as tar:
            tar.add(self.database_path, arcname=self.database_path.name)
        
        return str(archive_path)
    
    def decompress_tar_xz(self, archive_path: str, extract_to: Path) -> bool:
        """Decompress TAR.XZ archive."""
        print(f"Decompressing {archive_path}...")
        try:
            with tarfile.open(archive_path, 'r:xz') as tar:
                tar.extractall(extract_to)
            return True
        except Exception as e:
            print(f"❌ Decompression failed: {e}")
            return False
    
    def compress(self, method: str = "zip", compression_level: int = 6) -> str:
        """Compress the database using specified method."""
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Calculate and save checksums
        checksums = self.calculate_checksums()
        self.save_checksums(checksums)
        
        # Compress based on method
        if method.lower() == "zip":
            archive_path = self.compress_zip(compression_level)
        elif method.lower() == "tar.gz":
            archive_path = self.compress_tar_gz(compression_level)
        elif method.lower() == "tar.xz":
            archive_path = self.compress_tar_xz()
        else:
            raise ValueError(f"Unsupported compression method: {method}")
        
        # Get file sizes
        original_size = sum(f.stat().st_size for f in self.database_path.rglob('*') if f.is_file())
        compressed_size = Path(archive_path).stat().st_size
        compression_ratio = (1 - compressed_size / original_size) * 100
        
        print(f"\n📊 Compression Results:")
        print(f"   Original size: {original_size:,} bytes ({original_size / (1024*1024):.2f} MB)")
        print(f"   Compressed size: {compressed_size:,} bytes ({compressed_size / (1024*1024):.2f} MB)")
        print(f"   Compression ratio: {compression_ratio:.1f}%")
        print(f"   Archive saved to: {archive_path}")
        print(f"   Checksums saved to: {self.checksums_file}")
        
        return archive_path
    
    def decompress(self, archive_path: str, extract_to: Path = None, verify: bool = True) -> bool:
        """Decompress the database archive."""
        archive_path = Path(archive_path)
        if not archive_path.exists():
            raise FileNotFoundError(f"Archive not found: {archive_path}")
        
        if extract_to is None:
            extract_to = self.database_path.parent / "decompressed_database"
        
        extract_to.mkdir(parents=True, exist_ok=True)
        
        # Determine compression method and decompress
        if archive_path.suffix == '.zip':
            success = self.decompress_zip(str(archive_path), extract_to)
        elif archive_path.suffixes == ['.tar', '.gz']:
            success = self.decompress_tar_gz(str(archive_path), extract_to)
        elif archive_path.suffixes == ['.tar', '.xz']:
            success = self.decompress_tar_xz(str(archive_path), extract_to)
        else:
            raise ValueError(f"Unsupported archive format: {archive_path}")
        
        if not success:
            return False
        
        # Verify integrity if requested
        if verify:
            print("\n🔍 Verifying decompressed files...")
            if self.verify_integrity(extract_to):
                print("✅ All files verified successfully - zero data loss!")
                return True
            else:
                print("❌ Integrity verification failed!")
                return False
        
        return True

def main():
    parser = argparse.ArgumentParser(description="Compress and decompress database folder with zero data loss")
    parser.add_argument("action", choices=["compress", "decompress"], help="Action to perform")
    parser.add_argument("--database", "-d", default="database", help="Path to database folder")
    parser.add_argument("--output", "-o", help="Output path for compressed archive or decompressed folder")
    parser.add_argument("--method", "-m", choices=["zip", "tar.gz", "tar.xz"], default="zip", 
                       help="Compression method (default: zip)")
    parser.add_argument("--level", "-l", type=int, default=6, choices=range(1, 10),
                       help="Compression level 1-9 (default: 6)")
    parser.add_argument("--archive", "-a", help="Path to archive file (for decompression)")
    parser.add_argument("--no-verify", action="store_true", help="Skip integrity verification")
    
    args = parser.parse_args()
    
    try:
        compressor = DatabaseCompressor(args.database, args.output)
        
        if args.action == "compress":
            print(f"🗜️  Compressing database folder: {args.database}")
            archive_path = compressor.compress(args.method, args.level)
            print(f"\n✅ Compression completed successfully!")
            print(f"   Archive: {archive_path}")
            print(f"   Checksums: {compressor.checksums_file}")
            
        elif args.action == "decompress":
            if not args.archive:
                print("❌ Error: --archive argument required for decompression")
                sys.exit(1)
            
            print(f"📦 Decompressing archive: {args.archive}")
            success = compressor.decompress(args.archive, verify=not args.no_verify)
            
            if success:
                print(f"\n✅ Decompression completed successfully!")
            else:
                print(f"\n❌ Decompression failed!")
                sys.exit(1)
    
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

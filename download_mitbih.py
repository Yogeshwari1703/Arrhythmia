"""
Download REAL MIT-BIH Arrhythmia Database
FIXED VERSION with correct paths
"""

import os
import urllib.request
import zipfile
import wfdb

def download_mitbih_dataset():
    """Download complete MIT-BIH Arrhythmia Database"""
    
    print("="*60)
    print("DOWNLOADING REAL MIT-BIH ARRHYTHMIA DATABASE")
    print("="*60)
    
    # Get absolute path
    current_dir = os.getcwd()
    data_path = os.path.join(current_dir, 'data', 'mitbih')
    
    # Create data directory
    os.makedirs(data_path, exist_ok=True)
    
    print(f"\n📁 Saving to: {data_path}")
    
    # Method 1: Direct download (MORE RELIABLE)
    print("\n📥 Downloading via direct PhysioNet link...")
    
    url = "https://www.physionet.org/static/published-projects/mitdb/mit-bih-arrhythmia-database-1.0.0.zip"
    zip_path = os.path.join(current_dir, 'data', 'mitbih.zip')
    
    try:
        print("Downloading zip file (this will take 5-10 minutes)...")
        print("URL:", url)
        
        # Download with progress indicator
        def report_hook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = downloaded * 100 / total_size
                print(f"  Progress: {percent:.1f}%", end='\r')
        
        urllib.request.urlretrieve(url, zip_path, reporthook=report_hook)
        print("\n✅ Download complete!")
        
        # Extract
        print("Extracting files...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(os.path.join(current_dir, 'data'))
        print("✅ Extraction complete!")
        
        # Clean up
        os.remove(zip_path)
        print("✅ Cleanup complete!")
        
        # Count downloaded files
        mitbih_path = os.path.join(current_dir, 'data', 'mit-bih-arrhythmia-database-1.0.0')
        if os.path.exists(mitbih_path):
            files = os.listdir(mitbih_path)
            print(f"\n📊 Downloaded {len(files)} files")
            print("✅ SUCCESS: MIT-BIH database downloaded!")
            
            # Rename folder for easier access
            os.rename(mitbih_path, data_path)
            print(f"✅ Moved to: {data_path}")
            
            return True
            
    except Exception as e:
        print(f"\n❌ Direct download failed: {e}")
        print("\nTrying alternative method...")
        
        # Method 2: Using wfdb with correct path
        return download_with_wfdb(data_path)

def download_with_wfdb(data_path):
    """Alternative: Download using wfdb with correct paths"""
    print("\n📡 Downloading via wfdb...")
    
    records = [
        '100', '101', '102', '103', '104', '105', '106', '107', 
        '108', '109', '111', '112', '113', '114', '115', '116',
        '117', '118', '119', '121', '122', '123', '124', '200'
    ]
    
    downloaded = 0
    for record in records[:5]:  # Try first 5 records
        try:
            print(f"  Downloading {record}...")
            wfdb.dl_database('mitdb', record, data_path)
            print(f"  ✅ Downloaded {record}")
            downloaded += 1
        except Exception as e:
            print(f"  ❌ Failed to download {record}: {e}")
    
    print(f"\n📊 Downloaded {downloaded} records")
    return downloaded > 0

def manual_download_instructions():
    """Provide manual download instructions if automatic fails"""
    print("\n" + "="*60)
    print("MANUAL DOWNLOAD INSTRUCTIONS")
    print("="*60)
    print("\nIf automatic download fails, do this:")
    print("\n1. Open your browser and go to:")
    print("   https://physionet.org/content/mitdb/1.0.0/")
    print("\n2. Click 'Download ZIP'")
    print("\n3. Extract the ZIP file to:")
    print(f"   {os.path.join(os.getcwd(), 'data')}")
    print("\n4. Rename folder to 'mitbih'")
    print("\n✅ Then run the training script")

if __name__ == "__main__":
    success = download_mitbih_dataset()
    
    if success:
        print("\n" + "="*60)
        print("✅✅✅ SUCCESS! REAL MIT-BIH DATA DOWNLOADED ✅✅✅")
        print("="*60)
        print("\nNext step: Run train_real_model.py")
    else:
        manual_download_instructions()
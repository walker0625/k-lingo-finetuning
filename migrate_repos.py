import os
import shutil
from huggingface_hub import HfApi, snapshot_download, create_repo

# ==========================================
# [설정 영역]
# ==========================================
# TODO 본인의 Hugging Face 쓰기(Write) 권한 토큰 입력
HF_TOKEN = ''

# 2. 원본 리포지토리 정보
SOURCE_REPO_ID = "walker0625/k-lingo-finetuning"

# 3. 옮길 브랜치와 새로 만들 리포지토리 이름 매핑
# 형식: "브랜치명": "새로운_리포지토리명"
MIGRATION_MAP = {
    "level1": "walker0625/k-lingo-level1-lora",
    "level2": "walker0625/k-lingo-level2-lora",
    "level3": "walker0625/k-lingo-level3-lora",
}
# ==========================================

def migrate():
    if not HF_TOKEN or HF_TOKEN == "여기에_토큰을_입력하세요":
        print("❌ Error: Hugging Face API Token이 필요합니다.")
        return

    api = HfApi(token=HF_TOKEN)

    print(f"🚀 Migration Start: {SOURCE_REPO_ID} -> New Repos\n")

    for branch, new_repo_id in MIGRATION_MAP.items():
        print(f"-------------------------------------------------")
        print(f"📦 Processing Branch: '{branch}'")
        print(f"🎯 Target Repo: '{new_repo_id}'")
        
        try:
            # 1. 새로운 리포지토리 생성 (이미 있으면 건너뜀)
            print(f"   1. Creating repository '{new_repo_id}'...")
            create_repo(
                repo_id=new_repo_id, 
                token=HF_TOKEN, 
                exist_ok=True, 
                private=False # Public으로 할지 Private으로 할지 결정
            )
            
            # 2. 원본 브랜치 다운로드 (임시 폴더)
            print(f"   2. Downloading source files from branch '{branch}'...")
            local_path = snapshot_download(
                repo_id=SOURCE_REPO_ID,
                revision=branch,
                token=HF_TOKEN,
                ignore_patterns=[".gitattributes", ".git"] # 불필요한 파일 제외
            )
            print(f"      Downloaded to: {local_path}")

            # 3. 새로운 리포지토리에 업로드 (Main 브랜치로)
            print(f"   3. Uploading to '{new_repo_id}' (main branch)...")
            api.upload_folder(
                folder_path=local_path,
                repo_id=new_repo_id,
                repo_type="model",
                commit_message=f"Migrated from {SOURCE_REPO_ID}@{branch}"
            )
            
            print(f"✅ Success! Check: https://huggingface.co/{new_repo_id}")

        except Exception as e:
            print(f"❌ Failed to migrate {branch}: {e}")

    print("\n🎉 All migration tasks completed.")

if __name__ == "__main__":
    migrate()
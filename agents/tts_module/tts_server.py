import edge_tts
import uvicorn
import uuid
import os
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse

app = FastAPI()

# 容器内的缓存路径
CACHE_DIR = "/app/tts_cache"
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)


@app.get("/tts")
async def text_to_speech(text: str = Query(..., description="要转换的文字")):
    file_name = f"{uuid.uuid4()}.mp3"
    file_path = os.path.join(CACHE_DIR, file_name)

    # 保持之前自然的女声
    voice = "zh-CN-XiaoxiaoNeural"

    try:
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(file_path)
        return FileResponse(file_path, media_type="audio/mpeg")
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    # 必须监听 0.0.0.0
    uvicorn.run(app, host="0.0.0.0", port=5000)
import os
import json
import requests
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI()

@app.post("/api/")
async def process_questions(
    questions_txt: UploadFile = File(alias="questions.txt"),
    image_png: UploadFile = File(alias="image.png", default=None),
    data_csv: UploadFile = File(alias="data.csv", default=None)
):
    # Read questions.txt content
    try:
        content = await questions_txt.read()
        questions_text = content.decode('utf-8')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading questions.txt: {str(e)}")
    
    # Read other files if provided
    files_content = {"questions": questions_text}
    
    if image_png:
        try:
            image_content = await image_png.read()
            files_content["image"] = f"Image file provided: {image_png.filename}"
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error reading image.png: {str(e)}")
    
    if data_csv:
        try:
            csv_content = await data_csv.read()
            files_content["csv"] = csv_content.decode('utf-8')
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error reading data.csv: {str(e)}")
    
    # Get Gemini API key from environment
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY environment variable not set")
    
    # Prepare Gemini API request
    gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={api_key}"
    
    payload = {
        "contents": [{
            "parts": [{
                "text": f"Process the following data and respond with only a single valid JSON object or array, no additional text or explanation:\n\nQuestions: {questions_text}\n\n" + 
                       (f"CSV Data: {files_content.get('csv', '')}\n\n" if 'csv' in files_content else "") +
                       (f"{files_content.get('image', '')}\n\n" if 'image' in files_content else "")
            }]
        }],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 8192
        }
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    # Make request to Gemini API
    try:
        response = requests.post(gemini_url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        
        gemini_response = response.json()
        
        # Extract the generated text from Gemini response
        if "candidates" in gemini_response and len(gemini_response["candidates"]) > 0:
            generated_text = gemini_response["candidates"][0]["content"]["parts"][0]["text"]
            
            # Parse the JSON response from Gemini
            try:
                json_response = json.loads(generated_text.strip())
                return json_response
            except json.JSONDecodeError:
                # If JSON parsing fails, try to extract JSON from the text
                text = generated_text.strip()
                start_idx = text.find('{')
                end_idx = text.rfind('}') + 1
                if start_idx != -1 and end_idx > start_idx:
                    try:
                        json_response = json.loads(text[start_idx:end_idx])
                        return json_response
                    except json.JSONDecodeError:
                        pass
                
                # Check for array format
                start_idx = text.find('[')
                end_idx = text.rfind(']') + 1
                if start_idx != -1 and end_idx > start_idx:
                    try:
                        json_response = json.loads(text[start_idx:end_idx])
                        return json_response
                    except json.JSONDecodeError:
                        pass
                
                raise HTTPException(status_code=500, detail="Gemini did not return valid JSON")
        else:
            raise HTTPException(status_code=500, detail="No response from Gemini API")
            
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error calling Gemini API: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
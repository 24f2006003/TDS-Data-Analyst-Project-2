import os
import json
import google.generativeai as genai
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
    # Get Gemini API key from environment
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY environment variable not set")
    
    # Configure Gemini API
    genai.configure(api_key=api_key)
    
    # Read questions.txt content
    try:
        content = await questions_txt.read()
        questions_text = content.decode('utf-8')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading questions.txt: {str(e)}")
    
    # Read other files if provided
    prompt_parts = [f"Questions: {questions_text}"]
    
    if image_png:
        try:
            image_content = await image_png.read()
            prompt_parts.append(f"Image file provided: {image_png.filename}")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error reading image.png: {str(e)}")
    
    if data_csv:
        try:
            csv_content = await data_csv.read()
            csv_text = csv_content.decode('utf-8')
            prompt_parts.append(f"CSV Data: {csv_text}")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error reading data.csv: {str(e)}")
    
    # Create the full prompt with better context
    full_prompt = f"""You are a data analyst with web scraping and data analysis capabilities. 

For the following request, if it involves scraping data from a URL:
1. First scrape/fetch the data from the specified URL
2. Then perform the requested analysis on that data
3. Return only a valid JSON array or object with the exact results requested

If you need to create visualizations:
- Generate actual plots using the scraped data
- Return as base64 encoded PNG data URI format
- Keep file size under 100KB

IMPORTANT: 
- Actually fetch and use real data from URLs when specified
- Perform precise calculations on the actual data
- Return only valid JSON, no explanations or extra text
- Use exact numerical values, not approximations

Request to process:
{questions_text}

Additional data files:
""" + "\n".join([part for part in prompt_parts[1:] if part])
    
    # Generate response using Gemini with better configuration
    try:
        model = genai.GenerativeModel('gemini-2.0-flash-lite')
        
        # Configure generation parameters for better accuracy
        generation_config = genai.types.GenerationConfig(
            temperature=0.2,  # Slightly higher for web scraping tasks
            top_p=0.9,
            top_k=40,
            max_output_tokens=16384,  # More tokens for complex data analysis
        )
        
        response = model.generate_content(
            full_prompt,
            generation_config=generation_config
        )
        
        if not response.text:
            raise HTTPException(status_code=500, detail="No response from Gemini API")
        
        generated_text = response.text.strip()
        
        # Parse the JSON response from Gemini
        try:
            json_response = json.loads(generated_text)
            return json_response
        except json.JSONDecodeError:
            # If JSON parsing fails, try to extract JSON from the text
            text = generated_text
            
            # Try to find and extract JSON array first
            start_idx = text.find('[')
            end_idx = text.rfind(']') + 1
            if start_idx != -1 and end_idx > start_idx:
                try:
                    json_response = json.loads(text[start_idx:end_idx])
                    return json_response
                except json.JSONDecodeError:
                    pass
            
            # Then try JSON object
            start_idx = text.find('{')
            end_idx = text.rfind('}') + 1
            if start_idx != -1 and end_idx > start_idx:
                try:
                    json_response = json.loads(text[start_idx:end_idx])
                    return json_response
                except json.JSONDecodeError:
                    pass
            
            # Return the raw text wrapped in an error object if JSON parsing completely fails
            return {"error": "Could not parse JSON", "raw_response": generated_text}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calling Gemini API: {str(e)}")

@app.get("/")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
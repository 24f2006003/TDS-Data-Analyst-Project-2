import os
import json
import re
import requests
from bs4 import BeautifulSoup
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
    
    # Check if questions involve web scraping and fetch data if needed
    scraped_data = ""
    if "wikipedia.org" in questions_text or "scrape" in questions_text.lower():
        # Extract URL from the questions text
        url_match = re.search(r'https?://[^\s]+', questions_text)
        if url_match:
            url = url_match.group()
            try:
                # Fetch Wikipedia data
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find the main table (usually the first sortable table)
                table = soup.find('table', {'class': 'wikitable'})
                if table:
                    # Extract table data as CSV-like format
                    rows = []
                    headers = []
                    
                    # Get headers
                    header_row = table.find('tr')
                    if header_row:
                        for th in header_row.find_all(['th', 'td']):
                            headers.append(th.get_text().strip())
                    
                    # Get data rows
                    for row in table.find_all('tr')[1:]:  # Skip header row
                        cols = []
                        for td in row.find_all(['td', 'th']):
                            cols.append(td.get_text().strip())
                        if cols:
                            rows.append(cols)
                    
                    # Format as CSV-like text
                    if headers and rows:
                        scraped_data = f"Scraped Data from {url}:\n"
                        scraped_data += ",".join(headers) + "\n"
                        for row in rows[:100]:  # Increased limit for better analysis
                            scraped_data += ",".join(row) + "\n"
                        
            except Exception as e:
                scraped_data = f"Error scraping data: {str(e)}"
    
    # Read other files if provided
    prompt_parts = [f"Questions: {questions_text}"]
    
    if scraped_data:
        prompt_parts.append(f"Scraped Data: {scraped_data}")
    
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
    
    # Create the full prompt with strict formatting requirements
    full_prompt = f"""You are a data analyst with web scraping and data analysis capabilities.

CRITICAL INSTRUCTIONS:
- Respond with ONLY valid JSON - no markdown code blocks, no explanations, no extra text
- Do NOT wrap response in ```json``` or ``` blocks
- Return raw JSON array or object only
- For visualizations: return as base64 data URI in the JSON
- Ensure the entire response is complete valid JSON - do not truncate

For web scraping requests:
1. Use the provided scraped data below
2. Perform precise analysis on the actual data
3. Return exact numerical values, not approximations
4. For plots: generate actual matplotlib plots as base64 PNG data URI under 50KB (keep it smaller)
5. Make sure base64 strings are complete and not cut off

Request to process:
{questions_text}

""" + "\n".join([part for part in prompt_parts[1:] if part])
    
    # Generate response using Gemini with stricter configuration
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')  # Updated model
        
        # Configure for more consistent JSON output
        generation_config = genai.types.GenerationConfig(
            temperature=0.0,  # Very low for consistent formatting
            top_p=0.8,
            top_k=20,
            max_output_tokens=32768,  # Increased token limit for complete responses
        )
        
        response = model.generate_content(
            full_prompt,
            generation_config=generation_config
        )
        
        if not response.text:
            raise HTTPException(status_code=500, detail="No response from Gemini API")
        
        generated_text = response.text.strip()
        
        # Simply wrap the generated text in a JSON object and return
        return {
            "response": generated_text,
            "status": "success"
        }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calling Gemini API: {str(e)}")

@app.get("/")
async def health_check():
    return {"status": "healthy", "version": "1.1"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
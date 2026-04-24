import re
import asyncio

def remove_emojis():
    with open('README.md', 'r', encoding='utf-8') as f:
        content = f.read()

    # Emojis used in the README
    emojis = ['📋', '📁', '📊', '🚀', '🧠', '🌐', '📡', '🧪', '✅', '🐳', '🛠', '📝', '✨']
    for emoji in emojis:
        content = content.replace(emoji + ' ', '')
        content = content.replace(emoji, '')

    with open('README.md', 'w', encoding='utf-8') as f:
        f.write(content)

async def take_screenshots():
    from playwright.async_api import async_playwright
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        
        # Take screenshot of the Swagger UI
        try:
            await page.goto("http://localhost:8000/docs", wait_until="networkidle")
            # give it a bit more time to render
            await asyncio.sleep(2)
            await page.screenshot(path="docs_screenshot.png", full_page=True)
            print("Successfully took screenshot of Swagger UI")
        except Exception as e:
            print(f"Error taking docs screenshot: {e}")
            
        await browser.close()

if __name__ == '__main__':
    print("Removing emojis from README...")
    remove_emojis()
    print("Taking screenshots...")
    asyncio.run(take_screenshots())

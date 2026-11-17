"""Capture actual local Streamlit screens. Requires requirements-dev.txt and Chrome."""
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parents[1]

def main():
    target=ROOT/'docs'/'screenshots'
    target.mkdir(parents=True,exist_ok=True)
    with sync_playwright() as p:
        browser=p.chromium.launch(channel='chrome',headless=True)
        page=browser.new_page(viewport={'width':1440,'height':1050},device_scale_factor=1)
        errors=[]
        page.on('pageerror',lambda e:errors.append(str(e)))
        page.goto('http://localhost:8501',wait_until='networkidle')
        page.get_by_text('A clear view of observed trading behavior',exact=True).wait_for()
        sections=[('Overview','overview'),('Behavioral Analysis','behavioral_analysis'),('ML Predictions','ml_predictions'),('Trader Segmentation','trader_segmentation')]
        for label,slug in sections:
            page.get_by_text(label,exact=True).click()
            page.wait_for_timeout(1500)
            page.locator('[data-testid="stPlotlyChart"]').first.wait_for()
            if page.locator('[data-testid="stException"]').count():
                raise RuntimeError(page.locator('[data-testid="stException"]').inner_text())
            page.screenshot(path=str(target/f'{slug}.png'),full_page=True)
            print('Captured',slug)
        if errors: raise RuntimeError(errors)
        browser.close()

if __name__=='__main__': main()

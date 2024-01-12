from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time
from collections import defaultdict
import json
import re


driver = webdriver.Chrome(service= Service(ChromeDriverManager().install()))

url = "https://faq.bmwusa.com/s/?language=en_US"
driver.get(url)
driver.implicitly_wait(10)
time.sleep(2)

data = []
while True:
    try:
        learn_more_button = driver.find_element(By.XPATH, '/html/body/div[3]/div[2]/div/div/div[4]/div/div/c-scp-generic-article-list/div/div/div/button')
        learn_more_button.click()
    except:
        break

article_list = driver.find_element(By.XPATH, '/html/body/div[3]/div[2]/div/div/div[4]/div/div/c-scp-generic-article-list/div/div/div/div[2]')\
                        .find_elements(By.CSS_SELECTOR, '*')

for article in article_list:
    article.find_element(By.XPATH, f"child::*[1]").find_element(By.XPATH, f"child::*[1]")\
        .find_element(By.XPATH, f"child::*[2]").find_element(By.XPATH, f"child::*[1]").send_keys(Keys.ENTER)
        
time.sleep(5)

for article in article_list:
    try:
        question = article.find_element(By.XPATH, f"child::*[1]").find_element(By.XPATH, f"child::*[1]")\
                    .find_element(By.XPATH, f"child::*[2]").find_element(By.XPATH, f"child::*[1]").text

        answer = article.find_element(By.XPATH, f"child::*[1]").find_element(By.XPATH, f"child::*[1]")\
                    .find_element(By.XPATH, f"child::*[3]").find_element(By.XPATH, f"child::*[1]")\
                    .find_element(By.XPATH, f"child::*[1]").find_element(By.XPATH, f"child::*[1]")\
                    .find_element(By.XPATH, f"child::*[1]").find_element(By.XPATH, f"child::*[1]")\
                    .find_element(By.XPATH, f"child::*[1]").find_element(By.XPATH, f"child::*[1]")\
                    .find_element(By.XPATH, f"child::*[1]").find_element(By.XPATH, f"child::*[1]")\
                    .find_element(By.XPATH, f"child::*[1]").find_element(By.XPATH, f"child::*[1]")\
                    .find_element(By.XPATH, f"child::*[1]").find_element(By.XPATH, f"child::*[1]").text
        print(question)
        print(answer)
        data.append([question, answer])
    except:
        continue

print(len(data), 'data found')

with open('data.json', 'w') as f : 
	json.dump(data, f, indent=4)
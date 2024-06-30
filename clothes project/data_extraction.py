"""
Created on Sun Sep  3 17:04:14 2023

@author: safak
"""
# -*- coding:utf-8 -*-
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
import urllib.request
from urllib.request import urlopen
import os
import pandas as pd
import numpy as np

u=[]            
for i in range(24,40):
    
    url="https://www.trendyol.com/sr?q=erkek+t-shirt&qt=erkek+t-shirt&st=erkek+t-shirt&os=1&sk=1&pi="+str(i)
    #url="https://www.trendyol.com/sr?q=g%C3%B6mlek&qt=g%C3%B6mlek&st=g%C3%B6mlek&os=1&pi="+str(i)
    header={"User_Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"}
    r=requests.get(url,headers=header)
    soup=BeautifulSoup(r.content,"lxml")
    ürünler=soup.find_all("div",attrs={"class":"p-card-wrppr with-campaign-view"})
    u.append(ürünler)
     
def flatten_extend(matrix):
     flat_list = []
     for row in matrix:
         flat_list.extend(row)
     return flat_list
 
print(len(flatten_extend(u)))
a=flatten_extend(u)   

final_dict={}
result_dict={}
attr_list=[]
links=[]
sd=a
c=0
for ürün in sd:
    temp_list=[]
    i=ürün.find_all("div",attrs={"class":"p-card-chldrn-cntnr card-border"})
    i2=i[0].find_all("a")
    link_devam=i2[0].get("href")
    link_basi="https://www.trendyol.com/"
    link_tamami=link_basi+link_devam
    link_tamami=link_tamami.split("?")[0]
    links.append(link_tamami)
        
#her kıyafetin özelliklerini toplu olarak dict'e atma
for m in links:
    c+=1
    detay=requests.get(m)
    detay_soup=BeautifulSoup(detay.content, "html.parser")
    
    lk=detay_soup.find_all("main",{"id":"product-detail-app"})
    lk1=lk[0].find_all("div",{"class":"product-detail-container"})
    lk2=lk1[0].find_all("article",{"class":"pr-rnr-w"})
    lk3=lk2[0].find_all("div",{"class":"pr-rnr-cn gnr-cnt-br"})
    
    s6=detay_soup.find_all("ul",attrs={"class":"detail-attr-container"}) 
    l1=s6[0].find_all("li",attrs={"class":"detail-attr-item"})

    attr_list.append(s6)
    final_dict[c]=l1
    
    lk=detay_soup.find_all("main",{"id":"product-detail-app"})


    
    
#her kıyafetin görselinin hem linkini alıp hem de bilgisayara kaydetme
new_links=[]
u=1
for k in links:
    user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'
    headers={'User-Agent':user_agent,} 

    request=urllib.request.Request(k,None,headers) 
    response = urllib.request.urlopen(request)
    data = response.read() 

    page_soup=BeautifulSoup(data,"html.parser")
    s1=page_soup.find_all("div",{"class":"gallery-container"})
    s2=s1[0].find("div",{"class":"gallery-modal hidden"}) 
    s3=s2.find("div",{"class":"gallery-modal-content"})
    s4=s3.find_all("img")
    img_src=s4[0].get("src")
    
    filename=str(u)
    u+=1
    img_file=open("C:\\Users\\safak\\spyder projects\\clothes\\trendyol_imgs\\"+filename+".jpg","wb")
    img_file.write(urllib.request.urlopen(img_src,timeout=20).read())
    img_file.close()
        
    base_url=k
    extra_path = "/yorumlar?"
    full_url = base_url + extra_path
    new_links.append(full_url)
    
    
#her kıyafet özelliğini teker teker alıp dict'e kaydetme    
attribute_list=["Renk","Ortam", "Kol Boyu"]

rmv_img_list=[]
cnt=0
for i in final_dict:
    temp_dct={}
    cnt+=1
    lst=final_dict[i]
    for j in lst:
        st=""
        q1=j.find_all("span")
        
        if ((q1[1].text).startswith("Casual/G")):
            temp="Casual/Günlük"
        elif(q1[1].text=="Günlük" or q1[1].text=="GÃ¼nlÃ¼k"):
            temp="Casual/Günlük"
        else:
            temp=q1[1].text
        
        temp_dct[q1[0].text]= temp
    if len(temp_dct) < len(attribute_list):
        rmv_img_list.append(cnt)
        continue
    else:
        result_dict[cnt]=temp_dct
        
#sadece gerekli özellikleri ayıklama
last_dict={}
for j in result_dict.keys():
    tmp_d={}
    trs=result_dict[j]
    try:
        tmp_d[attribute_list[0]]=trs[attribute_list[0]]
        tmp_d[attribute_list[1]]=trs[attribute_list[1]]
        tmp_d[attribute_list[2]]=trs[attribute_list[2]]
        """tmp_d[attribute_list[3]]=trs[attribute_list[3]]
        tmp_d[attribute_list[4]]=trs[attribute_list[4]]
        tmp_d[attribute_list[5]]=trs[attribute_list[5]]"""
    except:
        rmv_img_list.append(j)
        continue
    last_dict[j]=tmp_d

last_new_links={}
#gereksiz fotoları silme
"""for m in rmv_img_list:
    os.remove("C:\\Users\\safak\\spyder projects\\clothes\\trendyol_imgs\\"+str(m)+".jpg")"""
    
for i in range(len(new_links)):
    if((i+1) not in rmv_img_list):
        last_new_links[i+1]=new_links[i]

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By

service=Service("C:\\Users\\safak\\spyder projects\\clothes\\chromedriver.exe")
driver=webdriver.Chrome(service=service)

like_val_list=[]
for i in last_new_links.keys():
    try:
        chrome_options=webdriver.ChromeOptions()
        chrome_options.add_argument("--incognito")
        chrome_options.add_argument("--headless")
        driver = webdriver.Chrome(options=chrome_options)
        driver.get(last_new_links[i])
        #driver.get("https://www.trendyol.com/youknitwear/sow-triko-t-shirt-p-757778695/yorumlar?")
        #driver.maximize_window()
        
        comment_button=driver.find_element(By.CLASS_NAME, "ps-stars__count")
        words = comment_button.text
        wrd = words.split("(")
        wrd2=wrd[1].split(")")[0]
        #print(wrd2)
        
        comment_button=driver.find_element(By.CSS_SELECTOR, "div.ps-ratings__count div")
        words2 = comment_button.text
        wrds = words2.split(" Değerlendirme")[0]
        #print(wrds)
        
        val=int(wrd2)/int(wrds)
        #print("Değerlendirme Puanı: {}".format(str(val)))
        like_val_list.append(val)
        
        #driver.quit()
    except:
        print("hata {}".format(i))
        rmv_img_list.append(i)
        continue
    

#gereksiz fotoları silme
for m in rmv_img_list:
    os.remove("C:\\Users\\safak\\spyder projects\\clothes\\trendyol_imgs\\"+str(m)+".jpg")
    try:
        last_dict.pop(m)
    except:
        continue
    
# Data Frame Oluşturma
df=pd.DataFrame(last_dict).T
df["Like"]=like_val_list
df["Key"]=df.index

df.to_csv("erkek_tshirt.csv",encoding='utf-8',index=False)

dd=pd.read_csv("erkek_tshirt.csv")


#ERKEK-> üst: [tişört], [gömlek], [ceket], [kazak], [sweatshirt]
#ERKEK-> alt: [şort], [eşofman], [pantolon]
#ERKEK-> [ayakkabı]

#KIZ -> üst: (tişört), [gömlek], [ceket], [kazak], crop
#KIZ -> pantolon, şort, etek
#KIZ -> ayakkabı
#KIZ -> komple: abiye, [elbise]


"""boy=0
kolboy=0
pacatipi=0
boyolcu=0
for i in result_dict:
    if "Boy" in result_dict[i]:
        boy+=1
    if "Boy / Ölçü" in result_dict[i]:
        boyolcu+=1
    if "Kol Boyu" in result_dict[i]:
        kolboy+=1
    if "Paça Tipi" in result_dict[i]:
        pacatipi+=1"""



"""for i in df[df.columns[4]]:
    df[df.columns[4]][i]=i+168

vertical_concat = pd.concat([dd, df], axis=0)
vertical_concat.to_csv("erkek_pantolon.csv",encoding='utf-8',index=False)"""




"""l=[]

for i in l[::-1]:
    dd=dd.drop(dd.index[i])

dd.to_csv("kadin_ayakkabi.csv",encoding='utf-8',index=False)"""















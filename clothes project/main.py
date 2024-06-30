from tkinter import *
from tkinter import messagebox
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
from tkinter import filedialog
from PIL import Image,ImageTk
import os
import tkinter as tk
from tkinter import ttk 

window=Tk()
window.title("Giriş Ekranı")
window.geometry("925x500+300+200")
window.configure(bg="#A0DEFF")
window.resizable(False,False)

sayac_e_gomlek=0
sayac_crop=0
sayac_elbise=0
sayac_k_ceket=0
sayac_e_esofman=0
sayac_e_ceket=0
sayac_e_ayakkabı=0
sayac_k_ayakkabı=0
sayac_e_sort=0
sayac_sweat=0
sayac_k_tshirt=0
sayac_abiye=0
sayac_k_sort=0
sayac_e_tshirt=0
sayac_e_kazak=0
sayac_e_pantolon=0
sayac_k_etek=0
sayac_k_gomlek=0
sayac_k_kazak=0
sayac_k_pantolon=0

tot_sayac=0

#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#PAGE 1
def login():
    username=user.get()
    password=code.get()
    
    if username=="admin" and password=="1907":
        window.destroy()

        form=tk.Tk()
        form.geometry("925x500+300+200")
        form.configure(bg="#A0DEFF")

        """frame=Frame(form2,width=75,height=500,bg="#222831")
        frame.place(x=0,y=0)"""
        
        def foto_yukle():
            from tkinter import filedialog
            import cv2
            from matplotlib import pyplot as plt
            import numpy as np
            from sklearn.preprocessing import MinMaxScaler
            import joblib
            import csv
            # Tkinter penceresini oluştur
            root = tk.Tk()
            root.title("Kıyafet Yükleme Ekranı")
            global sayac

            #root.geometry("925x500+300+200")
            root.configure(bg="#A0DEFF")
            font1=("Microsoft Yahei UI Light",9,"bold")
            font1=("Microsoft Yahei UI Light",15,"bold")
            font1=("Microsoft Yahei UI Light",18,"bold")
            root.geometry("925x500+300+200")  # Genişlik x Yükseklik

            # Pencereyi ekranda merkezlemek için gerekli kodlar

            


            #--------------------------------------------------------------
            #FONKSİYON KISMI (4.DERECE RENK MOMENTLERİ)
            """def ortalama(img):
                ort=0
                for i in range(img.shape[0]):
                    for j in range(img.shape[1]):
                        ort+=img[i,j]
                ort=ort/img.size
                return ort
            def efficient_mean(image):
                # Görüntüyü numpy dizisine dönüştür
                img_array = np.array(image)
                # Piksel değerlerini topla
                total = np.sum(img_array)
                # Piksel sayısını hesapla
                num_pixels = img_array.size
                # Ortalamayı hesapla
                mean = total / num_pixels
                return mean

            def standard_sapma(img):
                ort=ortalama(img)
                std_sapma=0
                for i in range(img.shape[0]):
                    for j in range(img.shape[1]):
                        std_sapma+=(img[i,j]-ort)**2
                std_sapma=std_sapma/img.size
                std_sapma=std_sapma**0.5
                return std_sapma
                
            def carpiklik(img):
                ort=ortalama(img)
                carpiklik = 0
                for i in range(img.shape[0]):
                    for j in range(img.shape[1]):
                        carpiklik += (img[i,j] - ort) ** 3
                
                carpiklik = carpiklik / (img.shape[1] * img.shape[0])
                return carpiklik ** (1/3)

            def basiklik(img):
                ort=ortalama(img)
                std_sapma=standard_sapma(img)
                basiklik=0
                for i in range(img.shape[0]):
                    for j in range(img.shape[1]):
                        basiklik+=(i-ort)**4
                basiklik=basiklik/img.size
                basiklik=basiklik/std_sapma**4 
                return basiklik"""



            def ortalama(img):
                ort=0
                for i in range(img.shape[0]):
                    for j in range(img.shape[1]):
                        ort+=img[i,j]
                ort=ort/img.size
                return ort

            def standard_sapma(img, ort):
                std_sapma=0
                for i in range(img.shape[0]):
                    for j in range(img.shape[1]):
                        std_sapma+=(img[i,j]-ort)**2
                std_sapma=std_sapma/img.size
                std_sapma=std_sapma**0.5
                return std_sapma
                
            def carpiklik(img, ort):
                carpiklik = 0
                for i in range(img.shape[0]):
                    for j in range(img.shape[1]):
                        carpiklik += (img[i,j] - ort) ** 3
                
                carpiklik = carpiklik / (img.shape[1] * img.shape[0])
                return carpiklik ** (1/3)

            def basiklik(img, ort):
                std_sapma=standard_sapma(img, ort)
                basiklik=0
                for i in range(img.shape[0]):
                    for j in range(img.shape[1]):
                        basiklik+=(i-ort)**4
                basiklik=basiklik/img.size
                basiklik=basiklik/std_sapma**4 
                return basiklik


            def min_max_norm(df):
                df=df.to_numpy().reshape(-1,1)
                scaler = MinMaxScaler()
                scaler.fit(df)
                nw=scaler.transform(df)
                return nw
            #--------------------------------------------------------------



            def moment_al(img_path):
                print("Çalışıyor 000")
                
                global new_path
                new_path=img_path.split("/")[-1]
                img=cv2.imread(new_path)


                print("Çalışıyor 111")
                if img is not None:  # Görüntü başarıyla yüklendi ise
                    # Görüntüyü yeniden boyutlandır
                    resized_img = cv2.resize(img, (256, 256))  # new_width ve new_height pozitif tam sayılar olmalı

                    # İşlenmiş görüntüyü kullan
                    cv2.imshow('Yeniden Boyutlandırılmış Görüntü', resized_img)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                else:
                    print("Görüntü yüklenemedi!")

                print("Çalışıyor 222")
                mask=np.zeros(img.shape[:2],np.uint8)
                cv2.circle(mask,(50,50),20,(255,255,255),cv2.FILLED)
                mask=cv2.dilate(mask,None,iterations=1)

                masked_img=cv2.bitwise_and(img, img,mask=mask)

                _,thresh=cv2.threshold(masked_img,thresh=np.mean(img)*1.5,maxval=255,type=cv2.THRESH_BINARY)

                result_d=np.hstack((img,masked_img,thresh))

                print("Çalışıyor 333")
                

                """hist=cv2.calcHist(img,channels=[0],mask=None,histSize=[256],ranges=[0,256])

                plt.figure(),plt.plot(hist),plt.axis("off"),plt.title("Picture")
                plt.show()"""
                #tm=[]

                """for i in range(3):

                    #print("ort: {}, std: {}, car: {}, bas:{}".format(ortalama(img),standard_sapma(img),carpiklik(img),basiklik(img)))
                    temp_list=[ortalama(img)[i],standard_sapma(img)[i],carpiklik(img)[i],basiklik(img)[i]]
                    
                    tm.append(min_max_norm(pd.Series(temp_list)))"""

                renk_mom_df=pd.DataFrame(columns=['Ort_R', 'Std_R', 'Car_R', 'Bas_R', 'Ort_G','Std_G', 'Car_G', 'Bas_G', 'Ort_B', 'Std_B', 'Car_B', 'Bas_B'])
                print("Çalışıyor 444")

                """yeni_satir = {'Ort_R':ortalama(img)[0], 'Std_R':standard_sapma(img)[0], 'Car_R':carpiklik(img)[0], 'Bas_R':basiklik(img)[0], 
                            'Ort_G':ortalama(img)[1],'Std_G':standard_sapma(img)[1], 'Car_G':carpiklik(img)[1], 'Bas_G':basiklik(img)[1],
                                'Ort_B':ortalama(img)[2], 'Std_B':standard_sapma(img)[2], 'Car_B':carpiklik(img)[2], 'Bas_B':basiklik(img)[2]}"""
                # Önce ortalama hesapla
                ort = ortalama(img)

                # Diğer özellikleri hesapla
                std_sapma_deger = standard_sapma(img, ort)
                carpiklik_deger = carpiklik(img, ort)
                basiklik_deger = basiklik(img, ort)
                
                yeni_satir = {'Ort_R':ort[0], 'Std_R':std_sapma_deger[0], 'Car_R':carpiklik_deger[0], 'Bas_R':basiklik_deger[0], 
                          'Ort_G':ort[1],'Std_G':std_sapma_deger[1], 'Car_G':carpiklik_deger[1], 'Bas_G':basiklik_deger[1],
                            'Ort_B':ort[2], 'Std_B':std_sapma_deger[2], 'Car_B':carpiklik_deger[2], 'Bas_B':basiklik_deger[2]}
                """renk_mom_df["Ort_R"]=ortalama(img)[0]
                renk_mom_df["Ort_G"]=ortalama(img)[1]
                renk_mom_df["Ort_B"]=ortalama(img)[2]

                renk_mom_df["Std_R"]=standard_sapma(img)[0]
                renk_mom_df["Std_G"]=standard_sapma(img)[1]
                renk_mom_df["Std_B"]=standard_sapma(img)[2]

                renk_mom_df["Car_R"]=carpiklik(img)[0]
                renk_mom_df["Car_G"]=carpiklik(img)[1]
                renk_mom_df["Car_B"]=carpiklik(img)[2]

                renk_mom_df["Bas_R"]=basiklik(img)[0]
                renk_mom_df["Bas_G"]=basiklik(img)[1]
                renk_mom_df["Bas_B"]=basiklik(img)[2]"""

                renk_mom_df = pd.concat([renk_mom_df, pd.DataFrame([yeni_satir])], ignore_index=True)

                print("Çalışıyor 555")
                

                
                

               #print(renk_mom_df)

                #print(min_max_norm(renk_mom_df))
                global renk_df
                renk_df=pd.DataFrame(columns=['Ort_R', 'Std_R', 'Car_R', 'Bas_R', 'Ort_G','Std_G', 'Car_G', 'Bas_G', 'Ort_B', 'Std_B', 'Car_B', 'Bas_B'])
                """renk_df=pd.DataFrame(index=['Ort_R', 'Std_R', 'Car_R', 'Bas_R', 'Ort_G','Std_G', 'Car_G', 'Bas_G', 'Ort_B', 'Std_B', 'Car_B', 'Bas_B'],data=min_max_norm(renk_mom_df))
                renk_df=renk_df.transpose()"""
                scaled_renk=min_max_norm(renk_mom_df)
                
                yeni_satir_renk = {'Ort_R': scaled_renk[0], 'Std_R': scaled_renk[1], 'Car_R': scaled_renk[2], "Bas_R":scaled_renk[3],"Ort_G":scaled_renk[4],"Std_G":scaled_renk[5],"Car_G":scaled_renk[6],"Bas_G":scaled_renk[7],"Ort_B":scaled_renk[8],"Std_B":scaled_renk[9],"Car_B":scaled_renk[10],"Bas_B":scaled_renk[11]}
                
                # Seçilen değerleri DataFrame'e eklemek için concat kullanma

               
                renk_df = pd.concat([renk_df, pd.DataFrame([yeni_satir_renk])], ignore_index=True)
                #print(renk_scaled_df)
                #status_label.config(text="Yükleme tamamlandı!", fg="green")
                return renk_df
            #status_label = tk.Label(root, text="", fg="black")
            #Dosya Seçme
            def dosya_sec_ve_yazdir():
                
                dosya_yolu = filedialog.askopenfilename(initialdir="/", title="Bir dosya seçin")
                
                # Dosya yolu boş değilse ve dosya seçildiyse
                if dosya_yolu:
                    # Dosya yolunu Text bileşenine yazdır
                    text_box.delete(1.0, tk.END)  # Mevcut içeriği temizle
                    text_box.insert(tk.END, dosya_yolu)  # Dosya yolunu ekle
                    
                    moment_al(dosya_yolu)
                    
                    

            # Text bileşeni oluşturma
            text_box = tk.Text(root, height=1, width=50)
            text_box.pack(pady=20)

            # Dosya seçme ve yazdırma için bir düğme oluşturma
            dosya_sec_buton = tk.Button(root, text="Dosya Seç", command=dosya_sec_ve_yazdir)

            dosya_sec_buton.pack(pady=10)


            #Kıyafet Yükleme
            """def parametreli_buton_olustur(img_path):
                print("parametreli buton oluştur fonk çağrıldı")
                return lambda: moment_al(img_path) """
            """load_buton = tk.Button(root, text="Kıyafet Yükle", command=dosya_sec_ve_yazdir)
            load_buton.pack(pady=10)"""


            # Combobox seçim işlevi
            def combobox_sec(event):
                global secilen1 
                secilen1= combo1.get()
                global secilen2
                secilen2 = combo2.get()
                global secilen3
                secilen3 = combo3.get()
                global secilen4
                secilen4 = combo4.get()
                
                
                if secilen1=="Erkek":
                    combo2["values"]=["Gömlek","Kazak","Pantolon","Eşofman","Ceket","Şort","Tshirt","Ayakkabı","Sweat"]
                if secilen1=="Kadin":
                    combo2["values"]=["Etek","Gömlek","Kazak","Pantolon","Crop","Elbise","Abiye","Ceket","Şort","Tshirt","Ayakkabı"]
                
                if secilen2=="Gömlek":
                    combo3["values"]=['Ortam_Business', 'Ortam_Casual/Günlük', 'Ortam_Sportswear','Ortam_Young', 'Ortam_Şık / gece']
                if secilen2=="Kazak":
                    combo3['values']=['Ortam_Casual/Günlük', 'Ortam_Gündüz / Gece', 'Ortam_Stylish/Night','Ortam_Young', 'Ortam_Şık / gece']
                if secilen2=="Pantolon":
                    combo3['values']=['Ortam_Back to School', 'Ortam_Business', 'Ortam_Casual/Günlük','Ortam_Gündüz / Gece', 'Ortam_Party', 'Ortam_Sportswear','Ortam_Stylish/Night', 'Ortam_Young', 'Ortam_Şık / gece']
                if secilen2=="Etek":
                    combo3['values']=['Ortam_Beachwear', 'Ortam_Business', 'Ortam_Casual/Günlük','Ortam_Gündüz / Gece', 'Ortam_Sportswear', 'Ortam_Young','Ortam_Şık / gece']
                if secilen2=="Crop":
                    combo3["values"]=["Ortam_Casual/Günlük","Ortam_Sportswear"]
                if secilen2=="Elbise":
                    combo3["values"]=["Ortam_Beachwear","Ortam_Business","Ortam_Casual/Günlük","Ortam_Şık / gece"]
                if secilen2=="Abiye":
                    combo3["values"]=['Ortam_Casual/Günlük', 'Ortam_Mezuniyet/Balo', 'Ortam_Şık / gece']
                if secilen2=="Ceket"  and secilen1=="Erkek":
                    combo3["values"]=["Ortam_Business","Ortam_Casual/Günlük"]
                if secilen2=="Ceket" and secilen1=="Kadin":
                    combo3["values"]=["Ortam_Business","Ortam_Casual/Günlük","Ortam_Party"]
                if secilen2=="Şort" and secilen1=="Kadin":
                    combo3["values"]=['Ortam_Casual/Günlük', 'Ortam_Sportswear', 'Ortam_Şık / gece']
                if secilen2=="Şort" and secilen1=="Erkek":
                    combo3["values"]=['Ortam_Casual/Günlük', 'Ortam_Sportswear']
                if secilen2=="Eşofman":
                    combo3["values"]=["Ortam_Casual/Günlük","Ortam_Lounge/Home","Ortam_Sportswear"]
                if secilen2=="Tshirt" and secilen1=="Kadin":
                    combo3["values"]=['Ortam_Business', 'Ortam_Casual/Günlük', 'Ortam_Party','Ortam_Sportswear']
                if secilen2=="Tshirt" and secilen1=="Erkek":
                    combo3["values"]=["Ortam_Casual/Günlük","Ortam_Sportswear","Ortam_Party"]
                if secilen2=="Ayakkabı" and secilen1=="Kadin":
                    combo3["values"]=["Ortam_Casual/Günlük","Ortam_Sportswear","Ortam_Şık / gece"]

                if secilen2=="Ayakkabı" and secilen1=="Erkek":
                    combo3["values"]=["Ortam_Casual/Günlük","Ortam_Sportswear"]
                if secilen2=="Sweat":
                    combo3["values"]=['Ortam_Casual/Günlük', 'Ortam_Şık / gece']

                    
                




                if secilen2=="Gömlek":
                    combo4["values"]=['Kol Boyu_Kısa Kol', 'Kol Boyu_Uzun','Kol Boyu_Uzun Kol']
                if secilen2=="Kazak":
                    combo4['values']=['Kol Boyu_Kısa Kol', 'Kol Boyu_Uzun','Kol Boyu_Uzun Kol']
                if secilen2=="Pantolon":
                    combo4['values']=['Paça Tipi_Bol Paça', 'Paça Tipi_Boru Paça', 'Paça Tipi_Dar Paça','Paça Tipi_Düz Paça', 'Paça Tipi_Geniş Paça', 'Paça Tipi_Lastikli Paça','Paça Tipi_Regular']
                if secilen2=="Etek":
                    combo4["values"]=['Boy / Ölçü_Diz Boyu', 'Boy / Ölçü_Kısa','Boy / Ölçü_Maxi', 'Boy / Ölçü_Midi', 'Boy / Ölçü_Mini','Boy / Ölçü_Regular', 'Boy / Ölçü_Uzun']
                
                
                if secilen2=="Crop":
                    combo4["values"]=["Kol Boyu_Kısa Kol","Kol Boyu_Sıfır Kol","Kol Boyu_Uzun Kol"]
                if secilen2=="Elbise":
                    combo4["values"]=["Kol Boyu_3/4 kol","Kol Boyu_Kolsuz","Kol Boyu_Kısa Kol","Kol Boyu_Uzun Kol"]
                if secilen2=="Abiye":
                    combo4["values"]=['Kol Boyu_Kısa Kol', 'Kol Boyu_Sıfır Kol', 'Kol Boyu_Tek Kol','Kol Boyu_Uzun Kol']
                if secilen2=="Ceket"  and secilen1=="Erkek":
                    combo4["values"]=["Kol Boyu_Kısa Kol","Kol Boyu_Uzun Kol"]
                if secilen2=="Ceket" and secilen1=="Kadin":
                    combo4["values"]=["Kol Boyu_3/4 kol","Kol Boyu_Kısa Kol","Kol Boyu_Uzun Kol"]
                if secilen2=="Şort" and secilen1=="Kadin":
                    combo4["values"]=['Kol Boyu_Dar Paça', 'Kol Boyu_Geniş Paça', 'Kol Boyu_Kısa Paça']
                if secilen2=="Şort" and secilen1=="Erkek":
                    combo4["values"]=['Boy_Kısa', 'Boy_Midi','Boy_Uzun']
                if secilen2=="Eşofman":
                    combo4["values"]=["Paça Tipi_Boru Paça","Paça Tipi_Dar Paça","Paça Tipi_Geniş Paça","Paça Tipi_Lastikli Paça","Paça Tipi_Normal Paça"]
                if secilen2=="Tshirt" and secilen1=="Kadin":
                    combo4["values"]=['Kol Boyu_Kısa Kol', 'Kol Boyu_Sıfır Kol','Kol Boyu_Uzun Kol']
                if secilen2=="Tshirt" and secilen1=="Erkek":
                    combo4["values"]=['Boy_3/4 kol', 'Boy_Kisa Kol', 'Boy_Uzun Kol']
                if secilen2=="Ayakkabı" and secilen1=="Kadin":
                    combo4["values"]=["Topuk Boyu_Kısa Topuklu (1- 4 cm)","Topuk Boyu_Orta Topuklu (5-9 cm)","Topuk Boyu_Yüksek Topuklu (10 cm ve üzeri)"]

                if secilen2=="Ayakkabı" and secilen1=="Erkek":
                    combo4["values"]=["Topuk Boyu_Kısa Topuklu (1- 4 cm)"]
                if secilen2=="Sweat":
                    combo4["values"]=['Kol Boyu_Kısa Kol','Kol Boyu_Uzun Kol']




                # Seçilen değerleri DataFrame'e ekle
                """global selected_values
                selected_values = selected_values.append({'Combo1': secilen1, 'Combo2': secilen2, 'Combo3': secilen3}, ignore_index=True)"""


            """def on_button_click():
                # Girdi kutusundan metni alıp işlem yapabiliriz
                global user_input
                user_input = entry.get()
                print("Yüklemek istediğiniz kıyafet sayısını giriniz:", user_input)
            entry = tk.Entry(root, width=30)
            entry.pack(pady=10)

            # Buton oluşturalım ve tıklama olayını belirleyelim
            button = tk.Button(root, text="Gönder", command=on_button_click)
            button.pack()"""

            global pred_list
            global path_list
            #progress_bar = ttk.Progressbar(root, orient="horizontal", length=200, mode="indeterminate")


            pred_list=[]
            path_list=[]



            

            #selected_values = pd.DataFrame(columns=['Cinsiyet', 'Kategori', 'Ortam',"Kol Boyu - Paça Tipi"])
            def df_aktar():
                path_aktar="C:\\Users\\safak\\spyder projects\\clothes\\MODELS"
                yeni_satir = {'Cinsiyet': secilen1, 'Kategori': secilen2, 'Ortam': secilen3, "Kol Boyu - Paça Tipi":secilen4}
                
                # Seçilen değerleri DataFrame'e eklemek için concat kullanma

                selected_values = pd.DataFrame(columns=['Cinsiyet', 'Kategori', 'Ortam',"Kol Boyu - Paça Tipi"])
                global sayac_e_gomlek
                global sayac_crop
                global sayac_elbise
                global sayac_k_ceket
                global sayac_e_esofman
                global sayac_e_ceket
                global sayac_e_ayakkabı
                global sayac_k_ayakkabı
                global sayac_e_sort
                global sayac_sweat
                global sayac_k_tshirt
                global sayac_abiye
                global sayac_k_sort
                global sayac_e_tshirt
                global sayac_e_kazak
                global sayac_e_pantolon
                global sayac_k_etek
                global sayac_k_gomlek
                global sayac_k_kazak
                global sayac_k_pantolon
                
                
                global res_df
                selected_values = pd.concat([selected_values, pd.DataFrame([yeni_satir])], ignore_index=True,axis=0)
                
                print("renk")
                print(renk_df)
                #res_df=pd.DataFrame(columns=['Ort_R', 'Std_R', 'Car_R', 'Bas_R', 'Ort_G','Std_G', 'Car_G', 'Bas_G', 'Ort_B', 'Std_B', 'Car_B', 'Bas_B','Cinsiyet', 'Kategori', 'Ortam',"Kol Boyu - Paça Tipi"])
                
                res_df=pd.concat([renk_df,selected_values],axis=1,ignore_index=True)
                print("====== res df ======")
                print(res_df)
                # DataFrame'i konsola yazdır
                #sıralama_gömlek: 'Ortam_Business', 'Ortam_Casual/Günlük', 'Ortam_Sportswear','Ortam_Young', 'Ortam_Şık / gece', 'Kol Boyu_Kısa Kol', 'Kol Boyu_Uzun','Kol Boyu_Uzun Kol', 'Ort_R', 'Std_R', 'Car_R', 'Bas_R', 'Ort_G','Std_G', 'Car_G', 'Bas_G', 'Ort_B', 'Std_B', 'Car_B', 'Bas_B'
                res_df.columns=['Ort_R', 'Std_R', 'Car_R', 'Bas_R', 'Ort_G','Std_G', 'Car_G', 'Bas_G', 'Ort_B', 'Std_B', 'Car_B', 'Bas_B','Cinsiyet', 'Kategori', 'Ortam',"Kol Boyu - Paça Tipi"]
                
                
                
                if secilen1=="Erkek" and secilen2=="Gömlek":
                    
                    ortam_list=['Ortam_Business', 'Ortam_Casual/Günlük', 'Ortam_Sportswear','Ortam_Young', 'Ortam_Şık / gece']
                    kol_boyu_list=['Kol Boyu_Kısa Kol', 'Kol Boyu_Uzun','Kol Boyu_Uzun Kol']
                    
                    one_hot_ortam=[0]*len(ortam_list)
                    for i in range(len(ortam_list)):
                        if res_df["Ortam"].values[sayac_e_gomlek]==ortam_list[i]:
                            one_hot_ortam[i]=1
                    
                    one_hot_kol=[0]*len(kol_boyu_list)
                    for i in range(len(kol_boyu_list)):
                        if res_df["Kol Boyu - Paça Tipi"].values[sayac_e_gomlek]==kol_boyu_list[i]:
                            one_hot_kol[i]=1
                    pred_df_ort=pd.DataFrame(index=ortam_list,data=one_hot_ortam)
                    pred_df_ort=pred_df_ort.transpose()
                    print("=======")
                    print(pred_df_ort)
                    pred_df_kol=pd.DataFrame(index=kol_boyu_list,data=one_hot_kol)
                    pred_df_kol=pred_df_kol.transpose()
                    print("=======")
                    print(pred_df_kol)
                    print("===== renk df iç ======")
                    print(renk_df)
                    pred_df=pd.concat([pred_df_ort,pred_df_kol,renk_df],axis=1,ignore_index=True)
                    print("======= pred_df ========")
                    print(pred_df)
                    print("======")
                    

                    erkek_gomlek=joblib.load(path_aktar+"\\MODEL_erkek_gomlek.joblib")
                    erkek_gomlek_prediction=erkek_gomlek.predict(pred_df)
                    pred_list.append(erkek_gomlek_prediction)
                    path_list.append(new_path)
                    """predict_and_path={"Predict":erkek_gomlek.predict(pred_df),"Path":new_path}
                    res_df = pd.concat([pred_df, pd.DataFrame(predict_and_path)], ignore_index=True,axis=1)"""
                    

                    sayac_e_gomlek=sayac_e_gomlek+1
                    
                
                if secilen1=="Kadin" and secilen2=="Crop": #Cinsiyet ve kategori kontrol et. Tek cinsiyete özel bir kategori varsa cinsiyet kontrol etmene gerek yok. seçilen2 kategori,secilen1 cinsiyet
                
                    ortam_list=['Ortam_Casual/Günlük', 'Ortam_Sportswear'] #modelde kullandığınız ortam değişkenleri
                    
                    
                    kol_boyu_list=['Kol Boyu_Kısa Kol', 'Kol Boyu_Sıfır Kol','Kol Boyu_Uzun Kol']

                    one_hot_ortam=[0]*len(ortam_list)
                    for i in range(len(ortam_list)):
                        if res_df["Ortam"].values[sayac_crop]==ortam_list[i]:
                            one_hot_ortam[i]=1
                    
                    one_hot_kol=[0]*len(kol_boyu_list)
                    for i in range(len(kol_boyu_list)):
                        if res_df["Kol Boyu - Paça Tipi"].values[sayac_crop]==kol_boyu_list[i]:
                            one_hot_kol[i]=1
                    pred_df_ort=pd.DataFrame(index=ortam_list,data=one_hot_ortam)
                    pred_df_ort=pred_df_ort.transpose()
                    print("=======")
                    print(pred_df_ort)
                    pred_df_kol=pd.DataFrame(index=kol_boyu_list,data=one_hot_kol)
                    pred_df_kol=pred_df_kol.transpose()
                    print("=======")
                    print(pred_df_kol)
                    print("===== renk df iç ======")
                    print(renk_df)
                    pred_df=pd.concat([pred_df_ort,pred_df_kol,renk_df],axis=1,ignore_index=True)
                    print("======= pred_df ========")
                    print(pred_df)
                    print("======")


                    kadın_crop=joblib.load(path_aktar+"\\MODEL_kadin_crop.joblib")
                    kadın_crop_prediction=kadın_crop.predict(pred_df)
                    pred_list.append(kadın_crop_prediction)
                    path_list.append(new_path)

                    sayac_crop=sayac_crop+1

                if secilen1=="Kadin" and secilen2=="Elbise": #Cinsiyet ve kategori kontrol et. Tek cinsiyete özel bir kategori varsa cinsiyet kontrol etmene gerek yok. seçilen2 kategori,secilen1 cinsiyet
                    
                    ortam_list=['Ortam_Casual/Günlük', 'Ortam_Business','Ortam_Beachwear','Ortam_Şık / gece'] #modelde kullandığınız ortam değişkenleri
                    
                   
                    kol_boyu_list=['Kol Boyu_Kısa Kol', 'Kol Boyu_Kolsuz','Kol Boyu_Uzun Kol','Kol Boyu_3/4 kol'] #modelde kullandığınız kol boyu paça tipi değişkenleri
                

                    one_hot_ortam=[0]*len(ortam_list)
                    for i in range(len(ortam_list)):
                        if res_df["Ortam"].values[sayac_elbise]==ortam_list[i]:
                            one_hot_ortam[i]=1
                    
                    one_hot_kol=[0]*len(kol_boyu_list)
                    for i in range(len(kol_boyu_list)):
                        if res_df["Kol Boyu - Paça Tipi"].values[sayac_elbise]==kol_boyu_list[i]:
                            one_hot_kol[i]=1
                    pred_df_ort=pd.DataFrame(index=ortam_list,data=one_hot_ortam)
                    pred_df_ort=pred_df_ort.transpose()
                    print("=======")
                    print(pred_df_ort)
                    pred_df_kol=pd.DataFrame(index=kol_boyu_list,data=one_hot_kol)
                    pred_df_kol=pred_df_kol.transpose()
                    print("=======")
                    print(pred_df_kol)
                    print("===== renk df iç ======")
                    print(renk_df)
                    pred_df=pd.concat([pred_df_ort,pred_df_kol,renk_df],axis=1,ignore_index=True)
                    print("======= pred_df ========")
                    print(pred_df)
                    print("======")


                    kadın_elbise=joblib.load(path_aktar+"\\MODEL_kadin_elbise.joblib")
                    kadın_elbise_prediction=kadın_elbise.predict(pred_df)
                    pred_list.append(kadın_elbise_prediction)
                    path_list.append(new_path)
                    sayac_elbise=sayac_elbise+1

                if secilen1=="Kadin" and secilen2=="Ceket": #Cinsiyet ve kategori kontrol et. Tek cinsiyete özel bir kategori varsa cinsiyet kontrol etmene gerek yok. seçilen2 kategori,secilen1 cinsiyet
                    
                    ortam_list=['Ortam_Casual/Günlük', 'Ortam_Business','Ortam_Party'] #modelde kullandığınız ortam değişkenleri
                    
                   
                    kol_boyu_list=['Kol Boyu_Kısa Kol','Kol Boyu_Uzun Kol','Kol Boyu_3/4 kol'] #modelde kullandığınız kol boyu paça tipi değişkenleri

                    one_hot_ortam=[0]*len(ortam_list)
                    for i in range(len(ortam_list)):
                        if res_df["Ortam"].values[sayac_k_ceket]==ortam_list[i]:
                            one_hot_ortam[i]=1
                    
                    one_hot_kol=[0]*len(kol_boyu_list)
                    for i in range(len(kol_boyu_list)):
                        if res_df["Kol Boyu - Paça Tipi"].values[sayac_k_ceket]==kol_boyu_list[i]:
                            one_hot_kol[i]=1
                    pred_df_ort=pd.DataFrame(index=ortam_list,data=one_hot_ortam)
                    pred_df_ort=pred_df_ort.transpose()
                    print("=======")
                    print(pred_df_ort)
                    pred_df_kol=pd.DataFrame(index=kol_boyu_list,data=one_hot_kol)
                    pred_df_kol=pred_df_kol.transpose()
                    print("=======")
                    print(pred_df_kol)
                    print("===== renk df iç ======")
                    print(renk_df)
                    pred_df=pd.concat([pred_df_ort,pred_df_kol,renk_df],axis=1,ignore_index=True)
                    print("======= pred_df ========")
                    print(pred_df)
                    print("======")


                    kadın_ceket=joblib.load(path_aktar+"\\MODEL_kadin_ceket.joblib")
                    kadın_ceket_prediction=kadın_ceket.predict(pred_df)
                    pred_list.append(kadın_ceket_prediction)
                    path_list.append(new_path)
                    sayac_k_ceket=sayac_k_ceket+1
                

                if secilen1=="Erkek" and secilen2=="Eşofman": #Cinsiyet ve kategori kontrol et. Tek cinsiyete özel bir kategori varsa cinsiyet kontrol etmene gerek yok. seçilen2 kategori,secilen1 cinsiyet
                    
                    ortam_list=['Ortam_Casual/Günlük', 'Ortam_Lounge/Home','Ortam_Sportswear'] #modelde kullandığınız ortam değişkenleri
                    
                   
                    kol_boyu_list=['Paça Tipi_Boru Paça','Paça Tipi_Dar Paça','Paça Tipi_Geniş Paça','Paça Tipi_Lastikli Paça','Paça Tipi_Normal Paça'] #modelde kullandığınız kol boyu paça tipi değişkenleri

                    one_hot_ortam=[0]*len(ortam_list)
                    for i in range(len(ortam_list)):
                        if res_df["Ortam"].values[sayac_e_esofman]==ortam_list[i]:
                            one_hot_ortam[i]=1
                    
                    one_hot_kol=[0]*len(kol_boyu_list)
                    for i in range(len(kol_boyu_list)):
                        if res_df["Kol Boyu - Paça Tipi"].values[sayac_e_esofman]==kol_boyu_list[i]:
                            one_hot_kol[i]=1
                    pred_df_ort=pd.DataFrame(index=ortam_list,data=one_hot_ortam)
                    pred_df_ort=pred_df_ort.transpose()
                    print("=======")
                    print(pred_df_ort)
                    pred_df_kol=pd.DataFrame(index=kol_boyu_list,data=one_hot_kol)
                    pred_df_kol=pred_df_kol.transpose()
                    print("=======")
                    print(pred_df_kol)
                    print("===== renk df iç ======")
                    print(renk_df)
                    pred_df=pd.concat([pred_df_ort,pred_df_kol,renk_df],axis=1,ignore_index=True)
                    print("======= pred_df ========")
                    print(pred_df)
                    print("======")


                    erkek_esofman=joblib.load(path_aktar+"\\MODEL_erkek_esofman.joblib")
                    erkek_esofman_prediction=erkek_esofman.predict(pred_df)
                    pred_list.append(erkek_esofman_prediction)
                    path_list.append(new_path)
                    sayac_e_esofman=sayac_e_esofman+1

                if secilen1=="Erkek" and secilen2=="Ceket": #Cinsiyet ve kategori kontrol et. Tek cinsiyete özel bir kategori varsa cinsiyet kontrol etmene gerek yok. seçilen2 kategori,secilen1 cinsiyet
                    
                    ortam_list=['Ortam_Casual/Günlük', 'Ortam_Business'] #modelde kullandığınız ortam değişkenleri
                    
                   
                    kol_boyu_list=['Kol Boyu_Uzun Kol','Kol Boyu_Kısa Kol'] #modelde kullandığınız kol boyu paça tipi değişkenleri

                    one_hot_ortam=[0]*len(ortam_list)
                    for i in range(len(ortam_list)):
                        if res_df["Ortam"].values[sayac_e_ceket]==ortam_list[i]:
                            one_hot_ortam[i]=1
                    
                    one_hot_kol=[0]*len(kol_boyu_list)
                    for i in range(len(kol_boyu_list)):
                        if res_df["Kol Boyu - Paça Tipi"].values[sayac_e_ceket]==kol_boyu_list[i]:
                            one_hot_kol[i]=1
                    pred_df_ort=pd.DataFrame(index=ortam_list,data=one_hot_ortam)
                    pred_df_ort=pred_df_ort.transpose()
                    print("=======")
                    print(pred_df_ort)
                    pred_df_kol=pd.DataFrame(index=kol_boyu_list,data=one_hot_kol)
                    pred_df_kol=pred_df_kol.transpose()
                    print("=======")
                    print(pred_df_kol)
                    print("===== renk df iç ======")
                    print(renk_df)
                    pred_df=pd.concat([pred_df_ort,pred_df_kol,renk_df],axis=1,ignore_index=True)
                    print("======= pred_df ========")
                    print(pred_df)
                    print("======")


                    erkek_ceket=joblib.load(path_aktar+"\\MODEL_erkek_ceket.joblib")
                    erkek_ceket_prediction=erkek_ceket.predict(pred_df)
                    pred_list.append(erkek_ceket_prediction)
                    path_list.append(new_path)
                    sayac_e_ceket=sayac_e_ceket+1

                if secilen1=="Erkek" and secilen2=="Ayakkabı": #Cinsiyet ve kategori kontrol et. Tek cinsiyete özel bir kategori varsa cinsiyet kontrol etmene gerek yok. seçilen2 kategori,secilen1 cinsiyet
                    
                    ortam_list=['Ortam_Casual/Günlük', 'Ortam_Sportswear'] #modelde kullandığınız ortam değişkenleri
                    
                   
                    kol_boyu_list=['Topuk Boyu_Kısa Topuklu (1- 4 cm)'] #modelde kullandığınız kol boyu paça tipi değişkenleri

                    one_hot_ortam=[0]*len(ortam_list)
                    for i in range(len(ortam_list)):
                        if res_df["Ortam"].values[sayac_e_ayakkabı]==ortam_list[i]:
                            one_hot_ortam[i]=1
                    
                    one_hot_kol=[0]*len(kol_boyu_list)
                    for i in range(len(kol_boyu_list)):
                        if res_df["Kol Boyu - Paça Tipi"].values[sayac_e_ayakkabı]==kol_boyu_list[i]:
                            one_hot_kol[i]=1
                    pred_df_ort=pd.DataFrame(index=ortam_list,data=one_hot_ortam)
                    pred_df_ort=pred_df_ort.transpose()
                    print("=======")
                    print(pred_df_ort)
                    pred_df_kol=pd.DataFrame(index=kol_boyu_list,data=one_hot_kol)
                    pred_df_kol=pred_df_kol.transpose()
                    print("=======")
                    print(pred_df_kol)
                    print("===== renk df iç ======")
                    print(renk_df)
                    pred_df=pd.concat([pred_df_ort,pred_df_kol,renk_df],axis=1,ignore_index=True)
                    print("======= pred_df ========")
                    print(pred_df)
                    print("======")


                    erkek_ayakkabi=joblib.load(path_aktar+"\\MODEL_erkek_ayakkabi.joblib")
                    erkek_ayakkabi_prediction=erkek_ayakkabi.predict(pred_df)
                    pred_list.append(erkek_ayakkabi_prediction)
                    path_list.append(new_path)
                    sayac_e_ayakkabı=sayac_e_ayakkabı+1

                if secilen1=="Kadin" and secilen2=="Ayakkabı": #Cinsiyet ve kategori kontrol et. Tek cinsiyete özel bir kategori varsa cinsiyet kontrol etmene gerek yok. seçilen2 kategori,secilen1 cinsiyet
                    
                    ortam_list=['Ortam_Casual/Günlük', 'Ortam_Sportswear','Ortam_Şık / gece'] #modelde kullandığınız ortam değişkenleri
                    
                   
                    kol_boyu_list=['Topuk Boyu_Kısa Topuklu (1- 4 cm)','Topuk Boyu_Orta Topuklu (5-9 cm)','Topuk Boyu_Yüksek Topuklu (10 cm ve üzeri)'] #modelde kullandığınız kol boyu paça tipi değişkenleri

                    one_hot_ortam=[0]*len(ortam_list)
                    for i in range(len(ortam_list)):
                        if res_df["Ortam"].values[sayac_k_ayakkabı]==ortam_list[i]:
                            one_hot_ortam[i]=1
                    
                    one_hot_kol=[0]*len(kol_boyu_list)
                    for i in range(len(kol_boyu_list)):
                        if res_df["Kol Boyu - Paça Tipi"].values[sayac_k_ayakkabı]==kol_boyu_list[i]:
                            one_hot_kol[i]=1
                    pred_df_ort=pd.DataFrame(index=ortam_list,data=one_hot_ortam)
                    pred_df_ort=pred_df_ort.transpose()
                    print("=======")
                    print(pred_df_ort)
                    pred_df_kol=pd.DataFrame(index=kol_boyu_list,data=one_hot_kol)
                    pred_df_kol=pred_df_kol.transpose()
                    print("=======")
                    print(pred_df_kol)
                    print("===== renk df iç ======")
                    print(renk_df)
                    pred_df=pd.concat([pred_df_ort,pred_df_kol,renk_df],axis=1,ignore_index=True)
                    print("======= pred_df ========")
                    print(pred_df)
                    print("======")


                    kadın_ayakkabi=joblib.load(path_aktar+"\\MODEL_kadin_ayakkabi.joblib")
                    kadın_ayakkabi_prediction=kadın_ayakkabi.predict(pred_df)
                    pred_list.append(kadın_ayakkabi_prediction)
                    path_list.append(new_path)
                    sayac_k_ayakkabı=sayac_k_ayakkabı+1

                if secilen1=="Erkek" and secilen2=="Şort": #Cinsiyet ve kategori kontrol et. Tek cinsiyete özel bir kategori varsa cinsiyet kontrol etmene gerek yok. seçilen2 kategori,secilen1 cinsiyet
                    
                    ortam_list=['Ortam_Casual/Günlük', 'Ortam_Sportswear'] #modelde kullandığınız ortam değişkenleri
                   
                    kol_boyu_list=['Boy_Kısa', 'Boy_Midi','Boy_Uzun'] #modelde kullandığınız kol boyu paça tipi değişkenleri


                    one_hot_ortam=[0]*len(ortam_list)
                    for i in range(len(ortam_list)):
                        if res_df["Ortam"].values[sayac_e_sort]==ortam_list[i]:
                            one_hot_ortam[i]=1
                    
                    one_hot_kol=[0]*len(kol_boyu_list)
                    for i in range(len(kol_boyu_list)):
                        if res_df["Kol Boyu - Paça Tipi"].values[sayac_e_sort]==kol_boyu_list[i]:
                            one_hot_kol[i]=1
                    pred_df_ort=pd.DataFrame(index=ortam_list,data=one_hot_ortam)
                    pred_df_ort=pred_df_ort.transpose()
                    print("=======")
                    print(pred_df_ort)
                    pred_df_kol=pd.DataFrame(index=kol_boyu_list,data=one_hot_kol)
                    pred_df_kol=pred_df_kol.transpose()
                    print("=======")
                    print(pred_df_kol)
                    print("===== renk df iç ======")
                    print(renk_df)
                    pred_df=pd.concat([pred_df_ort,pred_df_kol,renk_df],axis=1,ignore_index=True)
                    print("======= pred_df ========")
                    print(pred_df)
                    print("======")


                    erkek_sort=joblib.load(path_aktar+"\\MODEL_erkek_sort.joblib")
                    erkek_sort_prediction=erkek_sort.predict(pred_df)
                    pred_list.append(erkek_sort_prediction)
                    path_list.append(new_path)
                    sayac_e_sort=sayac_e_sort+1
                

                if secilen1=="Erkek" and secilen2=="Sweat": #Cinsiyet ve kategori kontrol et. Tek cinsiyete özel bir kategori varsa cinsiyet kontrol etmene gerek yok. seçilen2 kategori,secilen1 cinsiyet
                    
                    ortam_list=['Ortam_Casual/Günlük', 'Ortam_Şık / gece'] #modelde kullandığınız ortam değişkenleri
                   
                    kol_boyu_list=['Kol Boyu_Kısa Kol', 'Kol Boyu_Uzun Kol'] #modelde kullandığınız kol boyu paça tipi değişkenleri


                    one_hot_ortam=[0]*len(ortam_list)
                    for i in range(len(ortam_list)):
                        if res_df["Ortam"].values[sayac_sweat]==ortam_list[i]:
                            one_hot_ortam[i]=1
                    
                    one_hot_kol=[0]*len(kol_boyu_list)
                    for i in range(len(kol_boyu_list)):
                        if res_df["Kol Boyu - Paça Tipi"].values[sayac_sweat]==kol_boyu_list[i]:
                            one_hot_kol[i]=1
                    pred_df_ort=pd.DataFrame(index=ortam_list,data=one_hot_ortam)
                    pred_df_ort=pred_df_ort.transpose()
                    print("=======")
                    print(pred_df_ort)
                    pred_df_kol=pd.DataFrame(index=kol_boyu_list,data=one_hot_kol)
                    pred_df_kol=pred_df_kol.transpose()
                    print("=======")
                    print(pred_df_kol)
                    print("===== renk df iç ======")
                    print(renk_df)
                    pred_df=pd.concat([pred_df_ort,pred_df_kol,renk_df],axis=1,ignore_index=True)
                    print("======= pred_df ========")
                    print(pred_df)
                    print("======")


                    erkek_sweat=joblib.load(path_aktar+"\\MODEL_erkek_sweat.joblib")
                    erkek_sweat_prediction=erkek_sweat.predict(pred_df)
                    pred_list.append(erkek_sweat_prediction)
                    path_list.append(new_path)
                    sayac_sweat=sayac_sweat+1

                
                if secilen1=="Kadin" and secilen2=="Tshirt": #Cinsiyet ve kategori kontrol et. Tek cinsiyete özel bir kategori varsa cinsiyet kontrol etmene gerek yok. seçilen2 kategori,secilen1 cinsiyet
                    
                    ortam_list=['Ortam_Casual/Günlük', 'Ortam_Party', 'Ortam_Business', 'Ortam_Sportswear'] #modelde kullandığınız ortam değişkenleri
                   
                    kol_boyu_list=['Kol Boyu_Kısa Kol', 'Kol Boyu_Uzun Kol', 'Kol Boyu_Sıfır Kol'] #modelde kullandığınız kol boyu paça tipi değişkenleri

                    one_hot_ortam=[0]*len(ortam_list)
                    for i in range(len(ortam_list)):
                        if res_df["Ortam"].values[sayac_k_tshirt]==ortam_list[i]:
                            one_hot_ortam[i]=1
                    
                    one_hot_kol=[0]*len(kol_boyu_list)
                    for i in range(len(kol_boyu_list)):
                        if res_df["Kol Boyu - Paça Tipi"].values[sayac_k_tshirt]==kol_boyu_list[i]:
                            one_hot_kol[i]=1
                    pred_df_ort=pd.DataFrame(index=ortam_list,data=one_hot_ortam)
                    pred_df_ort=pred_df_ort.transpose()
                    print("=======")
                    print(pred_df_ort)
                    pred_df_kol=pd.DataFrame(index=kol_boyu_list,data=one_hot_kol)
                    pred_df_kol=pred_df_kol.transpose()
                    print("=======")
                    print(pred_df_kol)
                    print("===== renk df iç ======")
                    print(renk_df)
                    pred_df=pd.concat([pred_df_ort,pred_df_kol,renk_df],axis=1,ignore_index=True)
                    print("======= pred_df ========")
                    print(pred_df)
                    print("======")


                    kadin_tshirt=joblib.load(path_aktar+"\\MODEL_kadin_tshirt.joblib")
                    kadin_tshirt_prediction=kadin_tshirt.predict(pred_df)
                    pred_list.append(kadin_tshirt_prediction)
                    path_list.append(new_path)
                    sayac_k_tshirt=sayac_k_tshirt+1

                if secilen1=="Kadin" and secilen2=="Abiye": #Cinsiyet ve kategori kontrol et. Tek cinsiyete özel bir kategori varsa cinsiyet kontrol etmene gerek yok. seçilen2 kategori,secilen1 cinsiyet
                    
                    ortam_list=['Ortam_Casual/Günlük', 'Ortam_Mezuniyet/Balo', 'Ortam_Şık / gece',] #modelde kullandığınız ortam değişkenleri
                   
                    kol_boyu_list=['Kol Boyu_Kısa Kol', 'Kol Boyu_Uzun Kol', 'Kol Boyu_Sıfır Kol', 'Kol Boyu_Tek Kol'] #modelde kullandığınız kol boyu paça tipi değişkenleri
                
                    one_hot_ortam=[0]*len(ortam_list)
                    for i in range(len(ortam_list)):
                        if res_df["Ortam"].values[sayac_abiye]==ortam_list[i]:
                            one_hot_ortam[i]=1
                    
                    one_hot_kol=[0]*len(kol_boyu_list)
                    for i in range(len(kol_boyu_list)):
                        if res_df["Kol Boyu - Paça Tipi"].values[sayac_abiye]==kol_boyu_list[i]:
                            one_hot_kol[i]=1
                    pred_df_ort=pd.DataFrame(index=ortam_list,data=one_hot_ortam)
                    pred_df_ort=pred_df_ort.transpose()
                    print("=======")
                    print(pred_df_ort)
                    pred_df_kol=pd.DataFrame(index=kol_boyu_list,data=one_hot_kol)
                    pred_df_kol=pred_df_kol.transpose()
                    print("=======")
                    print(pred_df_kol)
                    print("===== renk df iç ======")
                    print(renk_df)
                    pred_df=pd.concat([pred_df_ort,pred_df_kol,renk_df],axis=1,ignore_index=True)
                    print("======= pred_df ========")
                    print(pred_df)
                    print("======")


                    kadin_abiye=joblib.load(path_aktar+"\\MODEL_kadin_abiye.joblib")
                    kadin_abiye_prediction=kadin_abiye.predict(pred_df)
                    pred_list.append(kadin_abiye_prediction)
                    path_list.append(new_path)
                    sayac_abiye=sayac_abiye+1

                if secilen1=="Kadin" and secilen2=="Şort": #Cinsiyet ve kategori kontrol et. Tek cinsiyete özel bir kategori varsa cinsiyet kontrol etmene gerek yok. seçilen2 kategori,secilen1 cinsiyet
                    
                    ortam_list=['Ortam_Casual/Günlük', 'Ortam_Sportswear', 'Ortam_Şık / gece',] #modelde kullandığınız ortam değişkenleri
                   
                    kol_boyu_list=['Kol Boyu_Dar Paça', 'Kol Boyu_Geniş Paça', 'Kol Boyu_Kısa Paça',] #modelde kullandığınız kol boyu paça tipi değişkenleri
                
                    one_hot_ortam=[0]*len(ortam_list)
                    for i in range(len(ortam_list)):
                        if res_df["Ortam"].values[sayac_k_sort]==ortam_list[i]:
                            one_hot_ortam[i]=1
                    
                    one_hot_kol=[0]*len(kol_boyu_list)
                    for i in range(len(kol_boyu_list)):
                        if res_df["Kol Boyu - Paça Tipi"].values[sayac_k_sort]==kol_boyu_list[i]:
                            one_hot_kol[i]=1
                    pred_df_ort=pd.DataFrame(index=ortam_list,data=one_hot_ortam)
                    pred_df_ort=pred_df_ort.transpose()
                    print("=======")
                    print(pred_df_ort)
                    pred_df_kol=pd.DataFrame(index=kol_boyu_list,data=one_hot_kol)
                    pred_df_kol=pred_df_kol.transpose()
                    print("=======")
                    print(pred_df_kol)
                    print("===== renk df iç ======")
                    print(renk_df)
                    pred_df=pd.concat([pred_df_ort,pred_df_kol,renk_df],axis=1,ignore_index=True)
                    print("======= pred_df ========")
                    print(pred_df)
                    print("======")


                    kadin_sort=joblib.load(path_aktar+"\\MODEL_kadin_sort.joblib")
                    kadin_sort_prediction=kadin_sort.predict(pred_df)
                    pred_list.append(kadin_sort_prediction)
                    path_list.append(new_path)
                    sayac_k_sort=sayac_k_sort+1
                
                if secilen1=="Erkek" and secilen2=="Tshirt": #Cinsiyet ve kategori kontrol et. Tek cinsiyete özel bir kategori varsa cinsiyet kontrol etmene gerek yok. seçilen2 kategori,secilen1 cinsiyet
                    print(2)
                    ortam_list=['Ortam_Casual/Günlük', 'Ortam_Sportswear', 'Ortam_Party',] #modelde kullandığınız ortam değişkenleri
                   
                    kol_boyu_list=['Boy_3/4 kol', 'Boy_Kısa Kol', 'Boy_Uzun Kol',] #modelde kullandığınız kol boyu paça tipi değişkenleri

                    one_hot_ortam=[0]*len(ortam_list)
                    for i in range(len(ortam_list)):
                        if res_df["Ortam"].values[sayac_e_tshirt]==ortam_list[i]:
                            one_hot_ortam[i]=1
                    
                    one_hot_kol=[0]*len(kol_boyu_list)
                    for i in range(len(kol_boyu_list)):
                        if res_df["Kol Boyu - Paça Tipi"].values[sayac_e_tshirt]==kol_boyu_list[i]:
                            one_hot_kol[i]=1
                    pred_df_ort=pd.DataFrame(index=ortam_list,data=one_hot_ortam)
                    pred_df_ort=pred_df_ort.transpose()
                    print("=======")
                    print(pred_df_ort)
                    pred_df_kol=pd.DataFrame(index=kol_boyu_list,data=one_hot_kol)
                    pred_df_kol=pred_df_kol.transpose()
                    print("=======")
                    print(pred_df_kol)
                    print("===== renk df iç ======")
                    print(renk_df)
                    pred_df=pd.concat([pred_df_ort,pred_df_kol,renk_df],axis=1,ignore_index=True)
                    print("======= pred_df ========")
                    print(pred_df)
                    print("======")


                    erkek_tshirt=joblib.load(path_aktar+"\\MODEL_erkek_tshirt.joblib")
                    erkek_tshirt_prediction=erkek_tshirt.predict(pred_df)
                    pred_list.append(erkek_tshirt_prediction)
                    path_list.append(new_path)
                    sayac_e_tshirt=sayac_e_tshirt+1
                
                
                if secilen1=="Erkek" and secilen2=="Kazak": #Cinsiyet ve kategori kontrol et. Tek cinsiyete özel bir kategori varsa cinsiyet kontrol etmene gerek yok. seçilen2 kategori,secilen1 cinsiyet
                    print(2)
                    ortam_list=['Ortam_Casual/Günlük', 'Ortam_Gündüz / Gece', 'Ortam_Stylish/Night','Ortam_Young', 'Ortam_Şık / gece'] #modelde kullandığınız ortam değişkenleri
                   
                    kol_boyu_list=['Kol Boyu_Kısa Kol', 'Kol Boyu_Uzun','Kol Boyu_Uzun Kol'] #modelde kullandığınız kol boyu paça tipi değişkenleri

                    one_hot_ortam=[0]*len(ortam_list)
                    for i in range(len(ortam_list)):
                        if res_df["Ortam"].values[sayac_e_kazak]==ortam_list[i]:
                            one_hot_ortam[i]=1
                    
                    one_hot_kol=[0]*len(kol_boyu_list)
                    for i in range(len(kol_boyu_list)):
                        if res_df["Kol Boyu - Paça Tipi"].values[sayac_e_kazak]==kol_boyu_list[i]:
                            one_hot_kol[i]=1
                    pred_df_ort=pd.DataFrame(index=ortam_list,data=one_hot_ortam)
                    pred_df_ort=pred_df_ort.transpose()
                    print("=======")
                    print(pred_df_ort)
                    pred_df_kol=pd.DataFrame(index=kol_boyu_list,data=one_hot_kol)
                    pred_df_kol=pred_df_kol.transpose()
                    print("=======")
                    print(pred_df_kol)
                    print("===== renk df iç ======")
                    print(renk_df)
                    pred_df=pd.concat([pred_df_ort,pred_df_kol,renk_df],axis=1,ignore_index=True)
                    print("======= pred_df ========")
                    print(pred_df)
                    print("======")


                    erkek_kazak=joblib.load(path_aktar+"\\MODEL_erkek_kazak.joblib")
                    erkek_kazak_prediction=erkek_kazak.predict(pred_df)
                    pred_list.append(erkek_kazak_prediction)
                    path_list.append(new_path)
                    sayac_e_kazak=sayac_e_kazak+1
                    
                if secilen1=="Erkek" and secilen2=="Pantolon": #Cinsiyet ve kategori kontrol et. Tek cinsiyete özel bir kategori varsa cinsiyet kontrol etmene gerek yok. seçilen2 kategori,secilen1 cinsiyet
                    print(2)
                    ortam_list=['Ortam_Back to School', 'Ortam_Business', 'Ortam_Casual/Günlük','Ortam_Gündüz / Gece', 'Ortam_Party', 'Ortam_Sportswear','Ortam_Stylish/Night', 'Ortam_Young', 'Ortam_Şık / gece'] #modelde kullandığınız ortam değişkenleri
                   
                    kol_boyu_list=['Paça Tipi_Bol Paça', 'Paça Tipi_Boru Paça', 'Paça Tipi_Dar Paça','Paça Tipi_Düz Paça', 'Paça Tipi_Geniş Paça', 'Paça Tipi_Lastikli Paça','Paça Tipi_Regular'] #modelde kullandığınız kol boyu paça tipi değişkenleri

                    one_hot_ortam=[0]*len(ortam_list)
                    for i in range(len(ortam_list)):
                        if res_df["Ortam"].values[sayac_e_pantolon]==ortam_list[i]:
                            one_hot_ortam[i]=1
                    
                    one_hot_kol=[0]*len(kol_boyu_list)
                    for i in range(len(kol_boyu_list)):
                        if res_df["Kol Boyu - Paça Tipi"].values[sayac_e_pantolon]==kol_boyu_list[i]:
                            one_hot_kol[i]=1
                    pred_df_ort=pd.DataFrame(index=ortam_list,data=one_hot_ortam)
                    pred_df_ort=pred_df_ort.transpose()
                    print("=======")
                    print(pred_df_ort)
                    pred_df_kol=pd.DataFrame(index=kol_boyu_list,data=one_hot_kol)
                    pred_df_kol=pred_df_kol.transpose()
                    print("=======")
                    print(pred_df_kol)
                    print("===== renk df iç ======")
                    print(renk_df)
                    pred_df=pd.concat([pred_df_ort,pred_df_kol,renk_df],axis=1,ignore_index=True)
                    print("======= pred_df ========")
                    print(pred_df)
                    print("======")


                    erkek_pantolon=joblib.load(path_aktar+"\\MODEL_erkek_pantolon.joblib")
                    erkek_pantolon_prediction=erkek_pantolon.predict(pred_df)
                    pred_list.append(erkek_pantolon_prediction)
                    path_list.append(new_path)
                    sayac_e_pantolon=sayac_e_pantolon+1
                    
                if secilen1=="Kadin" and secilen2=="Etek": #Cinsiyet ve kategori kontrol et. Tek cinsiyete özel bir kategori varsa cinsiyet kontrol etmene gerek yok. seçilen2 kategori,secilen1 cinsiyet
                    
                    ortam_list=['Ortam_Beachwear', 'Ortam_Business', 'Ortam_Casual/Günlük','Ortam_Gündüz / Gece', 'Ortam_Sportswear', 'Ortam_Young','Ortam_Şık / gece'] #modelde kullandığınız ortam değişkenleri
                   
                    kol_boyu_list=[ 'Boy / Ölçü_Diz Boyu', 'Boy / Ölçü_Kısa','Boy / Ölçü_Maxi', 'Boy / Ölçü_Midi', 'Boy / Ölçü_Mini','Boy / Ölçü_Regular', 'Boy / Ölçü_Uzun'] #modelde kullandığınız kol boyu paça tipi değişkenleri
                
                    one_hot_ortam=[0]*len(ortam_list)
                    for i in range(len(ortam_list)):
                        if res_df["Ortam"].values[sayac_k_etek]==ortam_list[i]:
                            one_hot_ortam[i]=1
                    
                    one_hot_kol=[0]*len(kol_boyu_list)
                    for i in range(len(kol_boyu_list)):
                        if res_df["Kol Boyu - Paça Tipi"].values[sayac_k_etek]==kol_boyu_list[i]:
                            one_hot_kol[i]=1
                    pred_df_ort=pd.DataFrame(index=ortam_list,data=one_hot_ortam)
                    pred_df_ort=pred_df_ort.transpose()
                    print("=======")
                    print(pred_df_ort)
                    pred_df_kol=pd.DataFrame(index=kol_boyu_list,data=one_hot_kol)
                    pred_df_kol=pred_df_kol.transpose()
                    print("=======")
                    print(pred_df_kol)
                    print("===== renk df iç ======")
                    print(renk_df)
                    pred_df=pd.concat([pred_df_ort,pred_df_kol,renk_df],axis=1,ignore_index=True)
                    print("======= pred_df ========")
                    print(pred_df)
                    print("======")


                    kadin_etek=joblib.load(path_aktar+"\\MODEL_kadin_etek.joblib")
                    kadin_etek_prediction=kadin_etek.predict(pred_df)
                    pred_list.append(kadin_etek_prediction)
                    path_list.append(new_path)
                    sayac_k_etek=sayac_k_etek+1
                    
                if secilen1=="Kadin" and secilen2=="Gömlek": #Cinsiyet ve kategori kontrol et. Tek cinsiyete özel bir kategori varsa cinsiyet kontrol etmene gerek yok. seçilen2 kategori,secilen1 cinsiyet
                    
                    ortam_list=['Ortam_Business', 'Ortam_Casual/Günlük', 'Ortam_Sportswear','Ortam_Young', 'Ortam_Şık / gece'] #modelde kullandığınız ortam değişkenleri
                   
                    kol_boyu_list=['Kol Boyu_Kısa Kol', 'Kol Boyu_Uzun','Kol Boyu_Uzun Kol'] #modelde kullandığınız kol boyu paça tipi değişkenleri
                
                    one_hot_ortam=[0]*len(ortam_list)
                    for i in range(len(ortam_list)):
                        if res_df["Ortam"].values[sayac_k_gomlek]==ortam_list[i]:
                            one_hot_ortam[i]=1
                    
                    one_hot_kol=[0]*len(kol_boyu_list)
                    for i in range(len(kol_boyu_list)):
                        if res_df["Kol Boyu - Paça Tipi"].values[sayac_k_gomlek]==kol_boyu_list[i]:
                            one_hot_kol[i]=1
                    pred_df_ort=pd.DataFrame(index=ortam_list,data=one_hot_ortam)
                    pred_df_ort=pred_df_ort.transpose()
                    print("=======")
                    print(pred_df_ort)
                    pred_df_kol=pd.DataFrame(index=kol_boyu_list,data=one_hot_kol)
                    pred_df_kol=pred_df_kol.transpose()
                    print("=======")
                    print(pred_df_kol)
                    print("===== renk df iç ======")
                    print(renk_df)
                    pred_df=pd.concat([pred_df_ort,pred_df_kol,renk_df],axis=1,ignore_index=True)
                    print("======= pred_df ========")
                    print(pred_df)
                    print("======")


                    kadin_gomlek=joblib.load(path_aktar+"\\MODEL_kadin_gomlek.joblib")
                    kadin_gomlek_prediction=kadin_gomlek.predict(pred_df)
                    pred_list.append(kadin_gomlek_prediction)
                    path_list.append(new_path)
                    sayac_k_gomlek=sayac_k_gomlek+1
                    
                if secilen1=="Kadin" and secilen2=="Kazak": #Cinsiyet ve kategori kontrol et. Tek cinsiyete özel bir kategori varsa cinsiyet kontrol etmene gerek yok. seçilen2 kategori,secilen1 cinsiyet
                    
                    ortam_list=[ 'Ortam_Casual/Günlük', 'Ortam_Gündüz / Gece', 'Ortam_Stylish/Night','Ortam_Young', 'Ortam_Şık / gece'] #modelde kullandığınız ortam değişkenleri
                   
                    kol_boyu_list=['Kol Boyu_Kısa Kol', 'Kol Boyu_Uzun','Kol Boyu_Uzun Kol'] #modelde kullandığınız kol boyu paça tipi değişkenleri
                
                    one_hot_ortam=[0]*len(ortam_list)
                    for i in range(len(ortam_list)):
                        if res_df["Ortam"].values[sayac_k_kazak]==ortam_list[i]:
                            one_hot_ortam[i]=1
                    
                    one_hot_kol=[0]*len(kol_boyu_list)
                    for i in range(len(kol_boyu_list)):
                        if res_df["Kol Boyu - Paça Tipi"].values[sayac_k_kazak]==kol_boyu_list[i]:
                            one_hot_kol[i]=1
                    pred_df_ort=pd.DataFrame(index=ortam_list,data=one_hot_ortam)
                    pred_df_ort=pred_df_ort.transpose()
                    print("=======")
                    print(pred_df_ort)
                    pred_df_kol=pd.DataFrame(index=kol_boyu_list,data=one_hot_kol)
                    pred_df_kol=pred_df_kol.transpose()
                    print("=======")
                    print(pred_df_kol)
                    print("===== renk df iç ======")
                    print(renk_df)
                    pred_df=pd.concat([pred_df_ort,pred_df_kol,renk_df],axis=1,ignore_index=True)
                    print("======= pred_df ========")
                    print(pred_df)
                    print("======")


                    kadin_kazak=joblib.load(path_aktar+"\\MODEL_kadin_kazak.joblib")
                    kadin_kazak_prediction=kadin_kazak.predict(pred_df)
                    pred_list.append(kadin_kazak_prediction)
                    path_list.append(new_path)
                    sayac_k_kazak=sayac_k_kazak+1
                    
                if secilen1=="Kadin" and secilen2=="Pantolon": #Cinsiyet ve kategori kontrol et. Tek cinsiyete özel bir kategori varsa cinsiyet kontrol etmene gerek yok. seçilen2 kategori,secilen1 cinsiyet
                    
                    ortam_list=['Ortam_Back to School', 'Ortam_Business', 'Ortam_Casual/Günlük','Ortam_Gündüz / Gece', 'Ortam_Party', 'Ortam_Sportswear','Ortam_Stylish/Night', 'Ortam_Young', 'Ortam_Şık / gece'] #modelde kullandığınız ortam değişkenleri
                   
                    kol_boyu_list=['Paça Tipi_Bol Paça', 'Paça Tipi_Boru Paça', 'Paça Tipi_Dar Paça','Paça Tipi_Düz Paça', 'Paça Tipi_Geniş Paça', 'Paça Tipi_Lastikli Paça','Paça Tipi_Regular'] #modelde kullandığınız kol boyu paça tipi değişkenleri
                
                    one_hot_ortam=[0]*len(ortam_list)
                    for i in range(len(ortam_list)):
                        if res_df["Ortam"].values[sayac_k_pantolon]==ortam_list[i]:
                            one_hot_ortam[i]=1
                    
                    one_hot_kol=[0]*len(kol_boyu_list)
                    for i in range(len(kol_boyu_list)):
                        if res_df["Kol Boyu - Paça Tipi"].values[sayac_k_pantolon]==kol_boyu_list[i]:
                            one_hot_kol[i]=1
                    pred_df_ort=pd.DataFrame(index=ortam_list,data=one_hot_ortam)
                    pred_df_ort=pred_df_ort.transpose()
                    print("=======")
                    print(pred_df_ort)
                    pred_df_kol=pd.DataFrame(index=kol_boyu_list,data=one_hot_kol)
                    pred_df_kol=pred_df_kol.transpose()
                    print("=======")
                    print(pred_df_kol)
                    print("===== renk df iç ======")
                    print(renk_df)
                    pred_df=pd.concat([pred_df_ort,pred_df_kol,renk_df],axis=1,ignore_index=True)
                    print("======= pred_df ========")
                    print(pred_df)
                    print("======")


                    kadin_pantolon=joblib.load(path_aktar+"\\MODEL_kadin_pantolon.joblib")
                    kadin_pantolon_prediction=kadin_pantolon.predict(pred_df)
                    pred_list.append(kadin_pantolon_prediction)
                    path_list.append(new_path)
                    sayac_k_pantolon=sayac_k_pantolon+1
                

                    



                print(res_df)
                print("===pred_list===")
                print(pred_list)
                print("====path_list")
                print(path_list)

            # Combobox bölümü
            tk.Label(root, text="Cinsiyet").pack()
            combo1 = ttk.Combobox(root,values=["Erkek","Kadin"])
            combo1.pack(pady=5)

            combo1.bind("<<ComboboxSelected>>", combobox_sec)

            tk.Label(root, text="Kategori").pack()
            combo2 = ttk.Combobox(root,values=["Etek","Gömlek","Kazak","Pantolon"])
            combo2.pack(pady=5)
            combo2.bind("<<ComboboxSelected>>", combobox_sec)

            tk.Label(root, text="Ortam").pack()
            combo3 = ttk.Combobox(root, values=['Ortam_Beachwear', 'Ortam_Business', 'Ortam_Casual/Günlük','Ortam_Gündüz / Gece', 'Ortam_Sportswear', 'Ortam_Young','Ortam_Şık / gece','Ortam_Stylish/Night',"Ortam_party",'Ortam_Back to School'])
            combo3.pack(pady=5)
            combo3.bind("<<ComboboxSelected>>", combobox_sec)
            tk.Label(root, text="Kol Boyu - Paça Tipi").pack()
            combo4 = ttk.Combobox(root,values=['Kol Boyu_Kısa Kol', 'Kol Boyu_Uzun','Kol Boyu_Uzun Kol','Paça Tipi_Bol Paça', 'Paça Tipi_Boru Paça', 'Paça Tipi_Dar Paça','Paça Tipi_Düz Paça', 'Paça Tipi_Geniş Paça', 'Paça Tipi_Lastikli Paça','Paça Tipi_Regular','Boy / Ölçü_Diz Boyu', 'Boy / Ölçü_Kısa','Boy / Ölçü_Maxi', 'Boy / Ölçü_Midi', 'Boy / Ölçü_Mini','Boy / Ölçü_Regular', 'Boy / Ölçü_Uzun'])
            combo4.pack(pady=5)
            combo4.bind("<<ComboboxSelected>>", combobox_sec)

            # Buton bölümü
            sec_buton = tk.Button(root, text="Ekle", command=df_aktar)
            sec_buton.pack(pady=10)


            def kaydet():
                global tot_sayac

                save_df=pd.read_csv("app_database_1.csv",encoding="cp1252")
                res_df["Tahmin"]=pred_list
                res_df["Path"]=str(len(save_df)+2)+".jpg"
                res_df_2=res_df.iloc[tot_sayac]
                res_df_2=pd.DataFrame(res_df_2)
                res_df_2=res_df_2.transpose()
                res_df_2=res_df_2.iloc[:,12:]
                res_df_2.columns=['Cinsiyet', 'Kategori', 'Ortam',"Kol Boyu - Paça Tipi","Tahmin","Path"]
                # DataFrame'deki verileri CSV dosyasına ekleyelim
                with open('app_database_1.csv', 'a', encoding="utf-8", newline='') as f:
                    writer = csv.writer(f)
                    writer.writerows(res_df_2.values)
                tot_sayac=tot_sayac+1
            save_buton=tk.Button(root,text="Kaydet",command=kaydet)
            save_buton.pack(pady=10)

            def back():
                root.destroy()

            back_bt=tk.Button(root,text="GERİ",width=5,height=1,bg="#A0DEFF",activebackground="#A0DEFF",command=back)
            back_bt.place(x=5,y=5)

            # Pencereyi çalıştır
            root.mainloop()


        def oneri_al():
            
            data_dict={} #cinsiyet, mevsim, hava, ciddiyet, balo
            
            lb1=tk.Label(form,text="Cinsiyet",fg="#03AED2",bg="#A0DEFF",font=("Microsoft Yahei UI Light",18,"bold"))
            lb1.place(x=420,y=40)
            """lb2=tk.Label(form,text="Erkek",fg="black",bg="#A0DEFF",font=("Microsoft Yahei UI Light",13,"bold"))
            lb2.place(x=400,y=80)
            lb3=tk.Label(form,text="Kadın",fg="black",bg="#A0DEFF",font=("Microsoft Yahei UI Light",13,"bold"))
            lb3.place(x=490,y=80)
            
            var1 = StringVar(form,"Erkek")
            
            rb1=tk.Radiobutton(form,variable=var1,value="Erkek",bg="#A0DEFF",activebackground="#A0DEFF",font=("Microsoft Yahei UI Light",12,"bold"))
            rb1.place(x=416,y=105)
            rb2=tk.Radiobutton(form,variable=var1,value="Kadın",bg="#A0DEFF",activebackground="#A0DEFF",font=("Microsoft Yahei UI Light",12,"bold"))
            rb2.place(x=507,y=105)"""
            
            
            gender = tk.StringVar() 
            cb1 = ttk.Combobox(form, width = 27,textvariable = gender) 
            cb1['values'] = ("Erkek","Kadın")
            cb1.place(x=376,y=80)
            
            var1=tk.StringVar()
            var2=tk.StringVar()
            var3=tk.StringVar()
            varx=tk.StringVar()
            
            
            
            """def lbk():
                filename="C:\\Users\\safak\\spyder projects\\clothes\\Veri_Tabanı_erkek\\DATA_erkek_ceket\\erkek_ceket\\1.jpg"
                img=Image.open(filename).resize((300,300))
                img=ImageTk.PhotoImage(img)
                lbl.configure(image=img)
                lbl.image=img"""
            
            def evet():
                varx.set("Evet")
                
            def hayir():
                varx.set("Hayır")
                
                lb4=tk.Label(form,text="Mevsim",fg="#03AED2",bg="#A0DEFF",font=("Microsoft Yahei UI Light",18,"bold"))
                lb4.place(x=290,y=80)
                lb5=tk.Label(form,text="Hava Durumu",fg="#03AED2",bg="#A0DEFF",font=("Microsoft Yahei UI Light",18,"bold"))
                lb5.place(x=290,y=120)
                lb6=tk.Label(form,text="Etkinlik",fg="#03AED2",bg="#A0DEFF",font=("Microsoft Yahei UI Light",18,"bold"))
                lb6.place(x=290,y=160)
                
                
                
                rb2=tk.Radiobutton(form,variable=var1,value="Yaz",text="Yaz",bg="#A0DEFF",activebackground="#A0DEFF",font=("Microsoft Yahei UI Light",12,"bold"))
                rb2.place(x=470,y=85)
                rb3=tk.Radiobutton(form,variable=var1,value="Kış",text="Kış",bg="#A0DEFF",activebackground="#A0DEFF",font=("Microsoft Yahei UI Light",12,"bold"))
                rb3.place(x=540,y=85)
                
                rb4=tk.Radiobutton(form,variable=var2,value="İyi",text="İyi",bg="#A0DEFF",activebackground="#A0DEFF",font=("Microsoft Yahei UI Light",12,"bold"))
                rb4.place(x=470,y=125)
                rb5=tk.Radiobutton(form,variable=var2,value="Kötü",text="Kötü",bg="#A0DEFF",activebackground="#A0DEFF",font=("Microsoft Yahei UI Light",12,"bold"))
                rb5.place(x=540,y=125)
                
                rb6=tk.Radiobutton(form,variable=var3,value="Ciddi",text="Ciddi",bg="#A0DEFF",activebackground="#A0DEFF",font=("Microsoft Yahei UI Light",12,"bold"))
                rb6.place(x=470,y=165)
                rb7=tk.Radiobutton(form,variable=var3,value="Rahat",text="Rahat",bg="#A0DEFF",activebackground="#A0DEFF",font=("Microsoft Yahei UI Light",12,"bold"))
                rb7.place(x=540,y=165)
                
                lb2.destroy()
                rb.destroy()
                rb1.destroy()
            
            lb2=tk.Label(form,text="Mezuniyet/Balo/Düğün tarzı\nbir etkinliğe mi katılacaksınız?",fg="#03AED2",bg="#A0DEFF",font=("Microsoft Yahei UI Light",18,"bold"))
            rb=tk.Button(form,width=5,height=1,text="Evet",bg="#03AED2",activebackground="#A0DEFF",font=("Microsoft Yahei UI Light",13,"bold"),command=evet)
            rb1=tk.Button(form,width=5,height=1,text="Hayır",bg="#03AED2",activebackground="#A0DEFF",font=("Microsoft Yahei UI Light",13,"bold"),command=hayir)
            
            
            def cinsiyet_sec():
                if (gender.get()=="Kadın"):
                    
                    lb2.place(x=290,y=45)
                    rb.place(x=400,y=120)
                    rb1.place(x=470,y=120)
                
                
                if(gender.get()=="Erkek"):
                    lb4=tk.Label(form,text="Mevsim",fg="#03AED2",bg="#A0DEFF",font=("Microsoft Yahei UI Light",18,"bold"))
                    lb4.place(x=290,y=80)
                    lb5=tk.Label(form,text="Hava Durumu",fg="#03AED2",bg="#A0DEFF",font=("Microsoft Yahei UI Light",18,"bold"))
                    lb5.place(x=290,y=120)
                    lb6=tk.Label(form,text="Etkinlik",fg="#03AED2",bg="#A0DEFF",font=("Microsoft Yahei UI Light",18,"bold"))
                    lb6.place(x=290,y=160)
                    
                    
                    
                    rb2=tk.Radiobutton(form,variable=var1,value="Yaz",text="Yaz",bg="#A0DEFF",activebackground="#A0DEFF",font=("Microsoft Yahei UI Light",12,"bold"))
                    rb2.place(x=470,y=85)
                    rb3=tk.Radiobutton(form,variable=var1,value="Kış",text="Kış",bg="#A0DEFF",activebackground="#A0DEFF",font=("Microsoft Yahei UI Light",12,"bold"))
                    rb3.place(x=540,y=85)
                    
                    rb4=tk.Radiobutton(form,variable=var2,value="İyi",text="İyi",bg="#A0DEFF",activebackground="#A0DEFF",font=("Microsoft Yahei UI Light",12,"bold"))
                    rb4.place(x=470,y=125)
                    rb5=tk.Radiobutton(form,variable=var2,value="Kötü",text="Kötü",bg="#A0DEFF",activebackground="#A0DEFF",font=("Microsoft Yahei UI Light",12,"bold"))
                    rb5.place(x=540,y=125)
                    
                    rb6=tk.Radiobutton(form,variable=var3,value="Ciddi",text="Ciddi",bg="#A0DEFF",activebackground="#A0DEFF",font=("Microsoft Yahei UI Light",12,"bold"))
                    rb6.place(x=470,y=165)
                    rb7=tk.Radiobutton(form,variable=var3,value="Rahat",text="Rahat",bg="#A0DEFF",activebackground="#A0DEFF",font=("Microsoft Yahei UI Light",12,"bold"))
                    rb7.place(x=540,y=165)
                
                def oner():
                    data_dict["Cinsiyet"]=gender.get()
                    data_dict["Mevsim"]=var1.get()
                    data_dict["Hava"]=var2.get()
                    data_dict["Ciddiyet"]=var3.get()
                    data_dict["Balo"]=varx.get()
                
                    #----------------------
                    ciddi_ortam=["Business","Şık / Gece","Mezuniyet/Balo","Party","Stylish/Night","Back to School"]
                    rahat_ortam=["Casual/Günlük","Sportswear","Lounge/Home","Gündüz / Gece","Young","Beachwear",]
            
                    erkek_y_ust=["tshirt","gomlek"]
                    erkek_y_alt=["sort","esofman"]
                    erkek_k_ust=["kazak","sweatshirt","gomlek"]
                    erkek_k_alt=["pantolon","esofman"]
                    erkek_ceket=["ceket"]
            
                    kadin_y_alt=["sort","pantolon","etek"]
                    kadin_y_ust=["tshirt","gomlek","crop"]
                    kadin_k_alt=["pantolon","etek"]
                    kadin_k_ust=["kazak","gomlek"]
                    kadin_ceket=["ceket"]
                    kadin_ciddi=["elbise","abiye"]
            
                    c="erkek" #default
                    my_alt=[]
                    my_ust=[]
                    my_ceket=[]
            
                    path="C:\\Users\\safak\\spyder projects\\clothes\\Veri_Tabanı_"+c
            
                    inp_gender=data_dict["Cinsiyet"]
                    ustuste=0
                    if(inp_gender=="Erkek"):
                        c="erkek"
                        rts="Erkek"
                        path="C:\\Users\\safak\\spyder projects\\clothes\\Veri_Tabanı_"+rts
                        ext="Hayır"
                        if(ext=="Hayır"):
                            mevsim=data_dict["Mevsim"]
                            if(mevsim=="Yaz"):
                                my_alt=erkek_y_alt
                                my_ust=erkek_y_ust
                                hava=data_dict["Hava"]
                                if (hava=="İyi"):
                                    
                                    
                                    ustuste=0
                                else:
                                    ustuste=1
                                    
                            else:
                                my_alt=erkek_k_alt
                                my_ust=erkek_k_ust
                                hava=data_dict["Hava"]
                                if (hava=="İyi"):
                                    pass
                                else:
                                    my_ust.remove("gomlek")
                                    my_ceket=erkek_ceket
                    else:
                        c="kadin"
                        rts="Kadın"
                        path="C:\\Users\\safak\\spyder projects\\clothes\\Veri_Tabanı_"+rts
                        ext=data_dict["Balo"]
                        if (ext=="Evet"):
                            clot_clothes=[]
                            my_clot=kadin_ciddi
                            for i in my_clot:
                                clot_clothes.append(c+"_"+i)
                            print(clot_clothes)
                            
                            dct_clot={}
                            dct_clot_lk={}
                            for m in clot_clothes:  
                                df=pd.read_csv(path+"\\DATA_"+m+"\\"+m+".csv") 
                                temp_list=[]
                                like_list=[]
                                temp_list.append(df["Key"])
                                like_list.append(df["Like"])
                                dct_clot[m]=temp_list
                                dct_clot_lk[m]=like_list
                            
                        else:
                            mevsim=data_dict["Mevsim"]
                            if(mevsim=="Yaz"):
                                my_alt=kadin_y_alt
                                my_ust=kadin_y_ust
                                hava=data_dict["Hava"]
                                if (hava=="İyi"):
                                    my_ust.remove("gomlek")
                                    my_alt.remove("pantolon")
                                    ustuste=0
                                else:
                                    ustuste=1
                                    
                            else:
                                my_alt=kadin_k_alt
                                my_ust=kadin_k_ust
                                hava=data_dict["Hava"]
                                if (hava=="İyi"):
                                    pass
                                else:
                                    my_alt.remove("etek")
                                    my_ust.remove("gomlek")
                                    my_ceket=kadin_ceket
                    if(ext=="Hayır"):       
                        print(my_alt)
                        print(my_ust)
                        print(my_ceket)
                        
                        print("**********************************")
                        
                        alt_clothes=[]
                        ust_clothes=[]
                        ceket_clothes=[]
                        
                        for i in my_alt:
                            alt_clothes.append(c+"_"+i)
                        
                        for i in my_ust:
                            ust_clothes.append(c+"_"+i)
                            
                        for i in my_ceket:
                            ceket_clothes.append(c+"_"+i)
                            
                        print(ust_clothes)
                        print(alt_clothes)
                        print(ceket_clothes)
                                    
                        
                        
                        #*************************************************************
                        
                        ortam=data_dict["Ciddiyet"]
                        if(ortam=="Ciddi"):
                            ort_cat=ciddi_ortam
                        else:
                            ort_cat=rahat_ortam
                        
                           
                        dct_ust={}
                        dct_ust_lk={} 
                        dct_alt={}
                        dct_alt_lk={} 
                        dct_ceket={}
                        dct_ceket_lk={}
                        for m in ust_clothes:  
                            df=pd.read_csv(path+"\\DATA_"+m+"\\"+m+".csv")
                            
                            final_ortam=[]
                            for i in df["Ortam"].unique():
                                if i in ort_cat:
                                    final_ortam.append(i)
                            temp_list=[]
                            like_list=[]
                            for j in final_ortam:
                                temp_list.append(df.loc[df["Ortam"]==j]["Key"])
                                like_list.append(df.loc[df["Ortam"]==j]["Like"])
                            dct_ust[m]=temp_list
                            dct_ust_lk[m]=like_list
                            
                        for m in alt_clothes:  
                            df=pd.read_csv(path+"\\DATA_"+m+"\\"+m+".csv")
                            
                            final_ortam=[]
                            for i in df["Ortam"].unique():
                                if i in ort_cat:
                                    final_ortam.append(i)
                            temp_list=[]
                            like_list=[]
                            for j in final_ortam:
                                temp_list.append(df.loc[df["Ortam"]==j]["Key"])
                                like_list.append(df.loc[df["Ortam"]==j]["Like"])
                            dct_alt[m]=temp_list
                            dct_alt_lk[m]=like_list
                        
                        for m in ceket_clothes:  
                            df=pd.read_csv(path+"\\DATA_"+m+"\\"+m+".csv")
                            
                            final_ortam=[]
                            for i in df["Ortam"].unique():
                                if i in ort_cat:
                                    final_ortam.append(i)
                            temp_list=[]
                            like_list=[]
                            for j in final_ortam:
                                temp_list.append(df.loc[df["Ortam"]==j]["Key"])
                                like_list.append(df.loc[df["Ortam"]==j]["Like"])
                            dct_ceket[m]=temp_list
                            dct_ceket_lk[m]=like_list
            
            
            
                    #de=pd.read_csv("C:\\Users\\safak\\spyder projects\\clothes\\Veri_Tabanı_Erkek\\DATA_"+m+"erkek_tshirt.csv")
            
                    def fltn(dct):
                        lst_ust=[]
                        for i in dct:
                            for j in dct[i]:
                                tmp=j.values.tolist()
                                for m in tmp:
                                    lst_ust.append(m)
                        return lst_ust
                    x_ceket_dict={}
                    x_ust_dict={}
                    x_alt_dict={}
                    x_clot_dict={}
                    if (ext=="Hayır"):
                        #ÜST  
                        lst_ust=fltn(dct_ust)
                        lst_ust_lk=fltn(dct_ust_lk)
                        
                        
                        for i in range(len(lst_ust)):
                            x_ust_dict[lst_ust[i]]=lst_ust_lk[i]
                        
                        #ALT
                        lst_alt=fltn(dct_alt)
                        lst_alt_lk=fltn(dct_alt_lk)
                        
                        
                        for i in range(len(lst_alt)):
                            x_alt_dict[lst_alt[i]]=lst_alt_lk[i]
                        
                        #CEKET
                        lst_ceket=fltn(dct_ceket)
                        lst_ceket_lk=fltn(dct_ceket_lk)
                        
                        
                        for i in range(len(lst_ceket)):
                            x_ceket_dict[lst_ceket[i]]=lst_ceket_lk[i]
                        
                    else:
                        #BALO
                        lst_clot=fltn(dct_clot)
                        lst_clot_lk=fltn(dct_clot_lk)
                        
                        
                        for i in range(len(lst_clot)):
                            x_clot_dict[lst_clot[i]]=lst_clot_lk[i]
                            
                    #*********KOMBİNASYON************
                    import itertools
            
                    likes=[]
            
                    if(len(x_ceket_dict)==0):
                        comb = list(itertools.product(x_ust_dict.keys(), x_alt_dict.keys()))
                        for i in comb:
                            temp_value=x_ust_dict[i[0]]*x_alt_dict[i[1]]
                            likes.append(temp_value)
                        
                    if(len(x_ceket_dict)!=0):
                        comb=list(itertools.product(x_ust_dict.keys(), x_alt_dict.keys(), x_ceket_dict.keys()))
                        for i in comb:
                            temp_value=x_ust_dict[i[0]]*x_alt_dict[i[1]]*x_ceket_dict[i[2]]
                            likes.append(temp_value)
                            
                    if(ext!="Hayır"):
                        print(444)
                        comb=list(x_clot_dict.keys())
                        likes=list(x_clot_dict.values())
            
            
                    maxes=sorted(likes)[::-1]
                    maxes=maxes[0]
                    indexes=[]
                    indexes.append(likes.index(maxes))
                    """indexes.append(likes.index(maxes[1]))
                    indexes.append(likes.index(maxes[2]))"""
            
                    final_recommend=[]
                    final_recommend.append(comb[indexes[0]])
                    """final_recommend.append(comb[indexes[1]])
                    final_recommend.append(comb[indexes[2]])   """
            
                    print(final_recommend)
                    print(type(final_recommend))
                
                        
                    lo_dict={}
                    if(type(final_recommend[0]))!=int:
                        if len(final_recommend[0])==2:
                            for i in dct_ust.keys():
                                lts=os.listdir(path+"\\"+"DATA_"+i+"\\"+i)
                                lts=list(map(lambda x: x.replace(x, x.split(".jpg")[0]), lts))
                                if str(final_recommend[0][0]) in lts:
                                    lo_dict["üst"]=i
                                
                            for i in dct_alt.keys():
                                lts=os.listdir(path+"\\"+"DATA_"+i+"\\"+i)
                                lts=list(map(lambda x: x.replace(x, x.split(".jpg")[0]), lts))
                                if str(final_recommend[0][1]) in lts:
                                    lo_dict["alt"]=i
                        
                        elif len(final_recommend[0])==3:
                            for i in dct_ust.keys():
                                lts=os.listdir(path+"\\"+"DATA_"+i+"\\"+i)
                                lts=list(map(lambda x: x.replace(x, x.split(".jpg")[0]), lts))
                                if str(final_recommend[0][0]) in lts:
                                    lo_dict["üst"]=i
                            for i in dct_alt.keys():
                                lts=os.listdir(path+"\\"+"DATA_"+i+"\\"+i)
                                lts=list(map(lambda x: x.replace(x, x.split(".jpg")[0]), lts))
                                if str(final_recommend[0][1]) in lts:
                                    lo_dict["alt"]=i
                            for i in dct_ceket.keys():
                                lts=os.listdir(path+"\\"+"DATA_"+i+"\\"+i)
                                lts=list(map(lambda x: x.replace(x, x.split(".jpg")[0]), lts))
                                if str(final_recommend[0][2]) in lts:
                                    lo_dict["ceket"]=i
                                
                    else:
                        for i in dct_clot.keys():
                            lts=os.listdir(path+"\\"+"DATA_"+i+"\\"+i)
                            lts=list(map(lambda x: x.replace(x, x.split(".jpg")[0]), lts))
                            if str(final_recommend[0]) in lts:
                                lo_dict["balo"]=i
                    
                    #--- show recommend -----
                    if len(lo_dict)==2:
                        filename="C:\\Users\\safak\\spyder projects\\clothes\\Veri_Tabanı_"+gender.get()+"\\DATA_"+lo_dict["üst"]+"\\"+lo_dict["üst"]+"\\"+str(final_recommend[0][0])+".jpg"
                        img=Image.open(filename).resize((150,150))
                        img=ImageTk.PhotoImage(img)
                        lbk=tk.Label(form,image=img)
                        lbk.place(x=300,y=250)
                        lbk.image=img
                        
                        filename2="C:\\Users\\safak\\spyder projects\\clothes\\Veri_Tabanı_"+gender.get()+"\\DATA_"+lo_dict["alt"]+"\\"+lo_dict["alt"]+"\\"+str(final_recommend[0][1])+".jpg"
                        img2=Image.open(filename2).resize((150,150))
                        img2=ImageTk.PhotoImage(img2)
                        lbk2=tk.Label(form,image=img2)
                        lbk2.place(x=480,y=250)
                        lbk2.image=img2
                    
                        oner.destroy()
                        lb4.destroy()
                        lb5.destroy()
                        lb6.destroy()
                        rb2.destroy()
                        rb3.destroy()
                        rb4.destroy()
                        rb5.destroy()
                        rb6.destroy()
                        rb7.destroy()
                        
                    elif len(lo_dict)==3:
                        filename3="C:\\Users\\safak\\spyder projects\\clothes\\Veri_Tabanı_"+gender.get()+"\\DATA_"+lo_dict["üst"]+"\\"+lo_dict["üst"]+"\\"+str(final_recommend[0][0])+".jpg"
                        img3=Image.open(filename3).resize((150,150))
                        img3=ImageTk.PhotoImage(img3)
                        lbk3=tk.Label(form,image=img3)
                        lbk3.place(x=213,y=250)
                        lbk3.image=img3
                        
                        filename4="C:\\Users\\safak\\spyder projects\\clothes\\Veri_Tabanı_"+gender.get()+"\\DATA_"+lo_dict["alt"]+"\\"+lo_dict["alt"]+"\\"+str(final_recommend[0][1])+".jpg"
                        img4=Image.open(filename4).resize((150,150))
                        img4=ImageTk.PhotoImage(img4)
                        lbk4=tk.Label(form,image=img4)
                        lbk4.place(x=393,y=250)
                        lbk4.image=img4
                        
                        filename5="C:\\Users\\safak\\spyder projects\\clothes\\Veri_Tabanı_"+gender.get()+"\\DATA_"+lo_dict["ceket"]+"\\"+lo_dict["ceket"]+"\\"+str(final_recommend[0][2])+".jpg"
                        img5=Image.open(filename5).resize((150,150))
                        img5=ImageTk.PhotoImage(img5)
                        lbk5=tk.Label(form,image=img5)
                        lbk5.place(x=573,y=250)
                        lbk5.image=img5
                    
                        oner.destroy()
                        lb4.destroy()
                        lb5.destroy()
                        lb6.destroy()
                        rb2.destroy()
                        rb3.destroy()
                        rb4.destroy()
                        rb5.destroy()
                        rb6.destroy()
                        rb7.destroy()
                    
                    elif len(lo_dict)==1:
                        filename6="C:\\Users\\safak\\spyder projects\\clothes\\Veri_Tabanı_"+gender.get()+"\\DATA_"+lo_dict["balo"]+"\\"+lo_dict["balo"]+"\\"+str(final_recommend[0])+".jpg"
                        img6=Image.open(filename6).resize((150,150))
                        img6=ImageTk.PhotoImage(img6)
                        lbk6=tk.Label(form,image=img6)
                        lbk6.place(x=393,y=250)
                        lbk6.image=img6
                    
                        oner.destroy()
                        lb4.destroy()
                        lb5.destroy()
                        lb6.destroy()
                        rb2.destroy()
                        rb3.destroy()
                        rb4.destroy()
                        rb5.destroy()
                        rb6.destroy()
                        rb7.destroy()
                    
                    
                    """filename="C:\\Users\\safak\\spyder projects\\clothes\\Veri_Tabanı_erkek\\DATA_erkek_ceket\\erkek_ceket\\1.jpg"
                    img=Image.open(filename).resize((150,150))
                    img=ImageTk.PhotoImage(img)
                    lbk=tk.Label(form,image=img)
                    lbk.place(x=50,y=50)
                    lbk.image=img"""
                    
                    
                    
                    
                    
                    
                    
                lb1.destroy()
                cb1.destroy()
                cins_sec.destroy()
                
                oner=tk.Button(form,width=10,height=2,text="ÖNER",bg="#7469B6",activebackground="#A0DEFF",font=("Microsoft Yahei UI Light",18,"bold"),command=oner)
                oner.place(x=390,y=280)
                
            
            
            cins_sec=tk.Button(form,width=7,height=1,text="SEÇ",bg="#03AED2",activebackground="#A0DEFF",font=("Microsoft Yahei UI Light",18,"bold"),command=cinsiyet_sec)
            cins_sec.place(x=412,y=150)

            lb.destroy()
            foto_yukle_bt.destroy()
            oneri_al_bt.destroy()
            
                
                

        def cikis_yap():
            form.destroy()
            
        def acc_infos():
            def back():
                form2.destroy()
                
            """img_back=Image.open("back.png").resize((34,34))
            img_back=ImageTk.PhotoImage(img_back)"""
            
            form2=tk.Tk()
            form2.geometry("925x500+300+200")
            form2.configure(bg="#A0DEFF")
            
            def gardrop_ac():
                file_path = filedialog.askopenfilename(initialdir="C:\\Users\\safak\\spyder projects\\clothes\\Veri_Tabanı_Kadın")
                file=open(file_path,"r")
                file.close()
            
            frame=Frame(form2,width=445,height=500,bg="#50C4ED")
            frame.place(x=480,y=0)

            label1=tk.Label(form2,text="Kullanıcı Adı:",fg="white",bg="#50C4ED",font=("Microsoft Yahei UI Light",15,"bold"))
            label1.place(x=580,y=70)

            ls1=tk.Label(form2,text="admin",fg="#387ADF",bg="#50C4ED",font=("Microsoft Yahei UI Light",15,"bold"))
            ls1.place(x=710,y=70)

            label2=tk.Label(form2,text="Şifre:",fg="white",bg="#50C4ED",font=("Microsoft Yahei UI Light",15,"bold"))
            label2.place(x=580,y=110)

            ls2=tk.Label(form2,text="1907",fg="#387ADF",bg="#50C4ED",font=("Microsoft Yahei UI Light",15,"bold"))
            ls2.place(x=632,y=110)

            label3=tk.Label(form2,text="Mail:",fg="white",bg="#50C4ED",font=("Microsoft Yahei UI Light",15,"bold"))
            label3.place(x=580,y=150)

            ls3=tk.Label(form2,text="admin123@gmail.com",fg="#387ADF",bg="#50C4ED",font=("Microsoft Yahei UI Light",15,"bold"))
            ls3.place(x=630,y=150)

            gard_bt=tk.Button(form2,text="Gardrop\nGörüntüle",width=9,height=2,bg="#A0DEFF",activebackground="#A0DEFF",font=("Microsoft Yahei UI Light",12,"bold"),command=gardrop_ac)
            gard_bt.place(x=655,y=280)
            
            back_bt=tk.Button(form2,text="GERİ",width=5,height=1,bg="#A0DEFF",activebackground="#A0DEFF",command=back)
            back_bt.place(x=5,y=5)
            
            form2.mainloop()


        img_acc=Image.open("acc.png").resize((32,32))
        img_acc=ImageTk.PhotoImage(img_acc)

        img_set=Image.open("set.png").resize((30,30))
        img_set=ImageTk.PhotoImage(img_set)

        img_out=Image.open("out.png").resize((38,38))
        img_out=ImageTk.PhotoImage(img_out)

        acc_bt=tk.Button(form,width=30,height=30,bg="#A0DEFF",activebackground="#A0DEFF",image=img_acc,command=acc_infos)
        acc_bt.place(x=5,y=5)

        lb_infos=tk.Label(form,text="Hesap",fg="#03AED2",bg="#A0DEFF",font=("Microsoft Yahei UI Light",13,"bold"))
        lb_infos.place(x=43,y=8)

        set_bt=tk.Button(form,width=30,height=30,bg="#A0DEFF",activebackground="#A0DEFF",image=img_set)
        set_bt.place(x=5,y=45)

        lb_set=tk.Label(form,text="Ayarlar",fg="#03AED2",bg="#A0DEFF",font=("Microsoft Yahei UI Light",13,"bold"))
        lb_set.place(x=43,y=48)

        out_bt=tk.Button(form,width=30,height=30,bg="#A0DEFF",activebackground="#A0DEFF",image=img_out,command=cikis_yap)
        out_bt.place(x=5,y=85)

        lb_out=tk.Label(form,text="Çıkış Yap",fg="#03AED2",bg="#A0DEFF",font=("Microsoft Yahei UI Light",13,"bold"))
        lb_out.place(x=43,y=88)



        #Label(form2,image=img,bg="#A0DEFF").place(x=855,y=5)


        lb=tk.Label(form,text="Yapmak istediğiniz işlemi seçiniz.",fg="#03AED2",bg="#A0DEFF",font=("Microsoft Yahei UI Light",18,"bold"))
        lb.place(x=271,y=70)

        foto_yukle_bt=tk.Button(form,width=9,height=2,text="Fotoğraf\nYükle",bg="#03AED2",activebackground="#A0DEFF",font=("Microsoft Yahei UI Light",18,"bold"),command=foto_yukle)
        foto_yukle_bt.place(x=305,y=150)

        oneri_al_bt=tk.Button(form,width=9,height=2,text="Öneri\nAl",bg="#03AED2",activebackground="#A0DEFF",font=("Microsoft Yahei UI Light",18,"bold"),command=oneri_al)
        oneri_al_bt.place(x=475,y=150)


        form.mainloop()
    else:
        messagebox.showerror("Hata!","Kullanıcı adı veya şifre yanlış.")

img=PhotoImage(file="log.png")
Label(window,image=img,bg="#A0DEFF").place(x=50,y=70)

frame=Frame(window,width=350,height=350,bg="#A0DEFF")
frame.place(x=480,y=70)

heading=Label(frame,text="Sign in",fg="#03AED2",bg="#A0DEFF",font=("Microsoft Yahei UI Light",23,"bold"))
heading.place(x=100,y=5)
#*********************************
def on_enter(e):
    user.delete(0,"end")
    
def on_leave(e):
    name=user.get()
    if name=="":
        user.insert(0,"Kullanıcı Adı")

user=Entry(frame,width=25,fg="black",border=0,bg="#A0DEFF",font=("Microsoft Yahei UI Light",11))
user.place(x=30,y=80)
user.insert(0,"Kullanıcı Adı")
user.bind("<FocusIn>",on_enter)
user.bind("<FocusOut>",on_leave)

Frame(frame,width=295,height=2,bg="black").place(x=25,y=107)
#*********************************
def on_enter(e):
    code.delete(0,"end")
    code.config(show="*")
    
def on_leave(e):
    name=code.get()
    if name=="":
        code.insert(0,"Şifre")
        
code=Entry(frame,width=25,fg="black",border=0,bg="#A0DEFF",font=("Microsoft Yahei UI Light",11))
code.place(x=30,y=150)
code.insert(0,"Şifre")
code.bind("<FocusIn>",on_enter)
code.bind("<FocusOut>",on_leave)

Frame(frame,width=295,height=2,bg="black").place(x=25,y=177)
#*********************************
Button(frame,width=39,pady=7,text="GİRİŞ YAP",bg="#03AED2",fg="white",border=0,command=login).place(x=35,y=204)
label=Label(frame,text="Hesabınız yok mu?",fg="black",bg="#A0DEFF",font=("Microsoft Yahei UI Light",9))
label.place(x=75,y=250)

sign_up=Button(frame,width=8,text="KAYIT OL",border=0,bg="#A0DEFF",cursor="hand2",fg="#03AED2",font=("Microsoft Yahei UI Light",9,"bold"))
sign_up.place(x=195,y=250)

window.mainloop() 
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
























#Reading files
import os 
from io import StringIO
import glob
import csv
import re
import pandas as pd
import xlrd
import itertools
import xlsxwriter
import openpyxl
from openpyxl import load_workbook
from bs4 import BeautifulSoup

#selenium
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys 
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import StaleElementReferenceException

#Additional
import requests
import urllib.request
import math
import lxml.html
import time
import import_ipynb
import _io
import schedule
import datetime
from datetime import datetime, timedelta
#from Goszakup_model_V2_main import *
import model


def world():
    print('Hello World')

    
####
# Main module that parses the given website
####
def module_main(results_file,key_file):

    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_experimental_option('useAutomationExtension', False)
    chrome_options.add_argument("--start-maximized")
    chrome_options.add_argument('--lang=ru-ru') 
    chrome_options.add_argument("headless")
    chrome_options.add_argument('window-size=1920x1080')
    driver = webdriver.Chrome(options=chrome_options, desired_capabilities=chrome_options.to_capabilities(), executable_path=r'C:\Users\anurimanov\Tender Analytics\chromedriver.exe')
    
    yesterday = datetime.now() - timedelta(1)
    type(yesterday)                                                                                                                                                                                    
    timestr = datetime.strftime(yesterday, '%d.%m.%Y')
    
    past_date = datetime.now() - timedelta(10)
    type(yesterday)                                                                                                                                                                                    
    past = datetime.strftime(past_date, '%d.%m.%Y')



    ####
    # Status: The contract took place: 
    # Summ: from 1,500,000 KZT
    # Publishind since: 01.01.2019
    # Number of tender on the web site
    #####
    
    count_records = 50
    
    ####
    # Lot filters
    ####
    
    status_1 = 190
    status_2 = 220
    status_3 = 280
    status_4 = 240
    status_5 = 210

    ####
    # Purchase filters
    ####
    
    method_1 = 3
    method_2 = 2
    method_3 = 7
    method_4 = 32
    method_5 = 22
    method_6 = 124
    method_7 = 126
    method_8 = 128
    method_9 = 129
    method_10 = 130


    status_str = '&filter%5Bstatus%5D%5B%5D='
    method_str = '&filter%5Bmethod%5D%5B%5D='

    template_begin = 'https://goszakup.gov.kz/ru/search/lots?filter%5Bname%5D=&filter%5Bnumber%5D=&filter%5Bnumber_anno%5D=&filter%5Benstru%5D='+str(method_str)+str(method_1)+str(method_str)+str(method_2)+str(method_str)+str(method_3)+str(method_str)+str(method_4)+str(method_str)+str(method_5)+str(method_str)+str(method_6)+str(method_str)+str(method_7)+str(method_str)+str(method_8)+str(method_str)+str(method_9)+str(method_str)+str(method_10)+str(status_str)+str(status_1)+str(status_str)+str(status_2)+str(status_str)+str(status_3)+str(status_str)+str(status_4)+str(status_str)+str(status_5)+'&filter%5Bamount_from%5D=1500000&filter%5Bstart_date_from%5D=01.01.2019'+'&count_record='+str(count_records)+'&filter%5Benstru%5D=' 
    url = ''
    num_tenders = ''


    initial_page = 1

    url = template_begin + '&page=' + str(initial_page)
    driver.get(url)
    driver.maximize_window()
    time.sleep(5)

    ####
    # count of tenders:    
    ####
    num_tenders = driver.find_elements_by_css_selector('div.dataTables_info')[0].text

    num_tenders = (re.findall(r'\d+', num_tenders))[2]
    
    ####
    # count of pages:
    ####


    pages = math.ceil(int(num_tenders)/count_records)




    if driver.find_elements_by_css_selector('div.dataTables_info')[0].text == 'Показано c 1 по 0 из 0 записей':
        pages = 0
    
    ####
    # Main loop that goes every page in the website
    ####
    for page in range(1, pages+1): #for page in range(1, pages+1):

        # Number of tender on the website (-1):
        try:
            tenders = len(driver.find_elements_by_css_selector('table')[1].find_elements_by_css_selector('tr'))
        except:
            url = template_begin + '&page=' + str(page)
            driver.get(url)
            driver.maximize_window()
            try:
                tenders = len(driver.find_elements_by_css_selector('table')[1].find_elements_by_css_selector('tr'))
            except:
                url = template_begin + '&page=' + str(page)
                driver.get(url)
                driver.maximize_window()


        lot_text = ''
        url_winner = ''

        tender_number = '' 
        tender_name = '' 
        contractor = '' 
        description = '' 
        status = ''
        cost = ''
        tru_number = '' 
        
        ####
        # Inner loop that parses every tender in the website
        ####
        for t in range(1, tenders): 

            lot_text = driver.find_elements_by_css_selector('table')[1].find_elements_by_css_selector('tr')[t].text
            lot_number = lot_text[:lot_text.find(' ')]

            lot_text = lot_text[len(lot_text[:lot_text.find(' ')])+1:]


            tender_number = lot_text[:lot_text.find(' ')]
            lot_text = lot_text[len(tender_number)+1:]


            tender_name = lot_text[:lot_text.find('\n')]
            lot_text = lot_text[len(tender_name)+1:]

            contractor = lot_text[lot_text.find('Заказчик: ')+10:lot_text.find('\n')]
            lot_text = lot_text[len(contractor)+11:]

            description = lot_text[:lot_text.find('\n')]
            lot_text = lot_text[len(description)+1:]
            lot_text = lot_text[lot_text.find(' ')+1:]

            cost = re.findall(r"[-+]?\d*\.\d+|\d+", lot_text)
            separator = ''
            cost = (separator.join(cost))


            lot_text = lot_text.split(".", 1)[1]
            status = lot_text[3:]

  
            ####
            #Creating empty table
            ####
            table = {
                    'Lot Number' : lot_number,
                    'Tender Number': tender_number,
                    'Tender Name': tender_name,
                    'Contractor': contractor,
                    'Description': description,
                    'Cost': cost,
                    'Status': status,
                    'Date_open' : timestr
                    }

            panda_table = pd.DataFrame(table, index=[0])

            panda_table['text'] = panda_table['Description'] + ' ' + panda_table['Tender Name']
            
            unumber = panda_table['Tender Number'].item()
            unumber = unumber.partition('-')[0]
            panda_table['url'] = 'https://goszakup.gov.kz/ru/announce/index/' + unumber

            key = pd.read_excel(key_file, index_col = 0)


            ####
            # Saving new key files, they will be replace the main key file for stopping rule
            ####

            if t == 1 and page == initial_page:
                print('key updated')
                key_new = panda_table

            if t == 2 and page == initial_page:
                print('key updated')
                key_new = key_new.append(panda_table, ignore_index=True)

            if t == 3 and page == initial_page:
                print('key updated')
                key_new = key_new.append(panda_table, ignore_index=True)

            if t == 4 and page == initial_page:
                print('key updated')
                key_new = key_new.append(panda_table, ignore_index=True)
            if t == 5 and page == initial_page:
                print('key updated')
                key_new = key_new.append(panda_table, ignore_index=True)

                
            ####
            #Applying feature engineering for parsed raw data
            ####

            panda_table['text'] = panda_table['text'].apply(model.r_dup)
            panda_table['text'] = model.text_process(panda_table['text'])
            panda_table['text'] = panda_table['text'].apply(model.lemmatize)


            predict = model.pipeline.predict(panda_table['text'])
            df_prediction = pd.DataFrame(predict)
            df_prediction.columns = ['Results']
            
            panda_table.drop(['text'], axis=1, inplace = True)
            panda_table = pd.concat([panda_table, df_prediction],axis=1)

            #panda_table = panda_table.loc[panda_table.Results == 1].reset_index(drop=True) # write only relevants
            
            result_table = pd.read_excel(results_file, index_col = 0)
            writer = pd.ExcelWriter(results_file, engine = 'xlsxwriter', date_format = 'mmmm yyyy')
            result_table = result_table.append(panda_table, ignore_index = True)
            result_table.to_excel(writer, 'Markets')
            writer.save()

            ####
            #Given that we have something in our key file, check if it contains any lot number equal to lot number in our result_table.
            #If it contains, rewrite the key file, with the new keys and save result table
            #Stop execution
            ####
        if len(key)>0:
            key = key.assign(checked=key['Lot Number'].isin(result_table.tail(50)['Lot Number']))   
            if len(key.loc[key['checked'] == True])>0:
                key = pd.read_excel(results_file, index_col = 0)
                writer_key = pd.ExcelWriter(key_file, engine = 'xlsxwriter', date_format = 'mmmm yyyy')
                key = key_new
                key.to_excel(writer_key, 'Markets')
                writer_key.save()
                print('all tenders synced')
                
                result_table = result_table[result_table.Date_open > past]
                result_table.reset_index(drop=True, inplace=True)
            
            
                result_table['Date_open'] = result_table['Date_open'].astype(str)
                
                #last_table = result_table.loc[result_table['Date_open'] == timestr]
                save_path = r'C:\Users\anurimanov\KPMG\Aleksandrova, Nataliia - DnA - Hackaton\Output\Result_main'+'.xlsx'
                last_table.to_excel(save_path, engine='xlsxwriter')  
                
                break
                
                
            ####
            #If our key file is empty, after parsing the first page, write key file with exicting keys
            ####
        if len(key)==0 and t == 5 and page == initial_page+1:
            key = pd.read_excel(key_file, index_col = 0)
            key = key_new
            writer_key = pd.ExcelWriter(key_file, engine = 'xlsxwriter', date_format = 'mmmm yyyy') #uncomment
            key.to_excel(writer_key, 'Markets') #uncomment
            writer_key.save() 

            
        
            ####
            #If last page, stop execution
            #Write key and result files
            ####
        if page == pages: 
            print('execution finished')
            key = pd.read_excel(key_file, index_col = 0)
            writer_key = pd.ExcelWriter(key_file, engine = 'xlsxwriter', date_format = 'mmmm yyyy')
            key=key_new
            key_new.to_excel(writer_key, 'Markets')
            writer_key.save()
            

            
            result_table = result_table[result_table.Date_open > past]
            result_table.reset_index(drop=True, inplace=True)
            
            result_table['Date_open'] = result_table['Date_open'].astype(str)
            #last_table = result_table.loc[result_table['Date_open'] == timestr]
            save_path = r'C:\Users\anurimanov\KPMG\Aleksandrova, Nataliia - DnA - Hackaton\Output\Result_main'+'.xlsx'
            last_table.to_excel(save_path, engine='xlsxwriter', index=False)  

            
            break

        print(page)
        url = template_begin + '&page=' + str(page + 1)
        driver.get(url)
        driver.maximize_window()
        time.sleep(1)

    driver.quit()
    return module_main()
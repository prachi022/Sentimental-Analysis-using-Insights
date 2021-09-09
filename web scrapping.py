#Importing Libraries
import pickle           #for importing model and features
import time             #to handle time-related tasks
from time import sleep
import logging
"""
Python has a built-in module logging which allows writing status messages to a file or any other output streams.
The file can contain the information on which part of the code is executed and what problems have been arisen.
Logging is a means of tracking events that happen when some software runs. Logging is important for software developing, debugging and running. 
If you don’t have any logging record and your program crashes, there are very little chances that you detect the cause of the problem
"""
import numpy as np
from selenium import webdriver     #web instance
from webdriver_manager.chrome import ChromeDriverManager     #web driver for chrome
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
start_time = time.time()
#The time() function returns the number of seconds passed since epoch.
#epouch defines the number times that the learning algorithm will work through the entire training dataset.

person = []
date = []
stars = []
review = []
sentiment = []

#defining export_data funtion to export the data in variables
def export_data():
    '''Exporting data'''
    dataframe1 = pd.DataFrame()         #create dataframe
    #filling the data in dataframe
    dataframe1["Person"] = person
    dataframe1["Date"] = date
    dataframe1["Stars"] = stars
    dataframe1["Reviews"] = review
    dataframe1["Sentiment"] = sentiment
    #saving the dataframe to a csv file
    dataframe1.to_csv('scrappedReviews.csv',index=False)
    
#defining function for Checking the sentiment of any review
def check_review(reviewtext):
    
    #Check the review is positive or negative
    
    #load the pickle file of model and features
    file = open("pickle_model.pkl", 'rb')
    pickle_model = pickle.load(file)
    file = open("features.pkl", 'rb')
    vocab = pickle.load(file)
    
    #reviewText(input of this function) has to be vectorised, that vectorizer is not saved yet
    #load the vectorize and call transform and then pass that to model preidctor
    #load it later
    transformer = TfidfTransformer()
    loaded_vec = CountVectorizer(decode_error="replace",vocabulary=vocab)
    vectorised_review = transformer.fit_transform(loaded_vec.fit_transform([reviewtext]))
    # Add code to test the sentiment of using both the model
    # 0 == negative   1 == positive
    out = pickle_model.predict(vectorised_review)
    return out[0]

#defining function for extracting the data from browser
def run_scraper():
    global person,review
    print("Starting Chrome:")
    #ChromeDriverManager() will manage the executable path
    browser = webdriver.Chrome(ChromeDriverManager().install())
    
    #Change these parameters on crash of code
    START_PAGE = 1
    END_PAGE = 251
    START_PRODUCT = 0
    
    URL =' https://www.etsy.com/in-en/c/jewelry-and-accessories?ref=pagination&page={}'
    try:
        #Count for every page of website(multiple pages)
        for page in range(START_PAGE,END_PAGE):
            #for every page it changes the url (parameter page =1,2,3....250)
            URL = URL.format(page)
            #open the browser with url
            browser.get(URL)
            print("Scraping Page:",page)
            #path of the page
            PATH_1 = '/html/body/div[5]/div/div[1]/div/div[4]/div[2]/div[2]/div[3]/div/div/ul'
            items = browser.find_element_by_xpath(PATH_1)
            #every product in the page
            items = items.find_elements_by_tag_name('li')
            #total product on one page
            end_product = len(items)
            #Count for every product of the page
            for product in range(START_PRODUCT,end_product):
                print("     Scarping reviews for product",product+1)
                if product!= 51:
                    items[product].find_element_by_tag_name('a').click()
                else:
                    continue
                #switch the focus of driver to new tab
                windows = browser.window_handles
                #switching the browser to next window/page
                #root window is at index 0 so next window will be at index 1
                browser.switch_to.window(windows[1])
                try:
                    count_review = browser.find_element_by_xpath('//*[@id="reviews"]/div[2]/nav/ul')
                    count_review = count_review.find_elements_by_tag_name('li')
                    last = (count_review[-2].find_element_by_tag_name('a').text)
                    last = int(last[last.find('\n'):])
                except Exception:
                    count_review = []
                    print("     No Pagination in reviews:")
                if len(count_review)>2:
                    #count for every product review pagination
                    for clc in range(0,last+1):
                        try:
                            PATH_2 = '//*[@id="same-listing-reviews-panel"]/div'
                            count = browser.find_element_by_xpath(PATH_2)
                            #Number of review on any page
                            count = count.find_elements_by_class_name('wt-grid__item-xs-12')
                            for r1 in range(1,len(count)+1):
                                dat1 = browser.find_element_by_xpath(
                                            '//*[@id="same-listing-reviews-panel"]/div/div[{}]/div[1]/div[2]/p[1]'.format(
                                                r1)).text
                                if dat1[:dat1.find(',')-6] not in person:
                                    try:
                                        person.append(dat1[:dat1.find(',')-6])
                                        date.append(dat1[dat1.find(',')-6:])
                                    except Exception:
                                        person.append("Not Found")
                                        date.append("Not Found")
                                    try:
                                        stars.append(browser.find_element_by_xpath(
                                            '//*[@id="same-listing-reviews-panel"]/div/div[{}]/div[2]/div/div/div[1]/span/span[1]'.format(
                                                r1)).text[0])
                                    except Exception:
                                        stars.append("No stars")
                                    try:
                                        review.append(browser.find_element_by_xpath(
                                            '//*[@id="review-preview-toggle-{}"]'.format(r1-1)).text)
                                        sentiment.append(check_review(browser.find_element_by_xpath(
                                            '//*[@id="review-preview-toggle-{}"]'.format(r1-1)).text))
                                    except Exception:
                                        review.append(np.nan)
                                        sentiment.append(check_review("No Review"))
                        except Exception:
                            try:
                                count = browser.find_element_by_xpath('//*[@id="reviews"]/div[2]/div[2]')
                                count = count.find_elements_by_class_name('wt-grid__item-xs-12')
                                
                                for r2 in range(1,len(count)+1):
                                    dat1 = browser.find_element_by_xpath(
                                                '//*[@id="reviews"]/div[2]/div[2]/div[{}]/div[1]/p'.format(r2)).text
                                    if dat1[:dat1.find(',')-6] not in person:
                                        try:
                                            
                                            person.append(dat1[:dat1.find(',')-6])
                                            date.append(dat1[dat1.find(',')-6:])
                                        except Exception:
                                            person.append("Not Found")
                                            date.append("Not Found")
                                        try:
                                            stars.append(browser.find_element_by_xpath(
                                                '//*[@id="reviews"]/div[2]/div[2]/div[{}]/div[2]/div[1]/div[1]/div[1]/span/span[1]'.format(
                                                    r2)).text[0])
                                        except Exception:
                                            stars.append("No Stars")
                                        try:
                                            review.append(browser.find_element_by_xpath(
                                                '//*[@id="review-preview-toggle-{}"]'.format(
                                                    r2-1)).text)
                                            sentiment.append(check_review(
                                                browser.find_element_by_xpath(
                                                '//*[@id="review-preview-toggle-{}"]'.format(
                                                    r2-1)).text))
                                        except Exception:
                                            review.append(np.nan)
                                            sentiment.append(check_review(
                                                "No Review"))                                        
                            except Exception:
                                try:
                                    count = browser.find_element_by_xpath('//*[@id="reviews"]/div[2]/div[2]')
                                    count = count.find_elements_by_class_name('wt-grid__item-xs-12')
                                    
                                    for r3 in range(1,len(count)+1):
                                        dat1 = browser.find_element_by_xpath(
                                                    '//*[@id="same-listing-reviews-panel"]/div/div[{}]/div[1]/p'.format(r3)).text
                                        if dat1[:dat1.find(',')-6] not in person:
                                            try:
                                                person.append(dat1[:dat1.find(',')-6])
                                                date.append(dat1[dat1.find(',')-6:])
                                            except Exception:
                                                person.append("Not Found")
                                                date.append("Not Found")
                                            try:
                                                stars.append(browser.find_element_by_xpath(
                                                    '//*[@id="same-listing-reviews-panel"]/div/div[{}]/div[2]/div[1]/div[1]/div[1]/span/span[1]'.format(r3)).text[0])
                                            except Exception:
                                                stars.append("No Stars")
                                            try:
                                                review.append(browser.find_element_by_xpath(
                                                    '//*[@id="review-preview-toggle-{}"]'.format(r3-1)).text)
                                                sentiment.append(check_review(browser.find_element_by_xpath(
                                                    '//*[@id="review-preview-toggle-{}"]'.format(r3-1)).text))
                                            except Exception:
                                                review.append(np.nan)
                                                sentiment.append(check_review("No Review"))
                                except Exception:
                                    print("Error")
                                    continue
                        sleep(1)
                        count_review = browser.find_element_by_xpath('//*[@id="reviews"]/div[2]/nav/ul')
                        count_review = count_review.find_elements_by_tag_name('li')
                        sleep(1)
                        count_review[-1].find_element_by_tag_name('a').click()
                        sleep(1)
                browser.close()
                #swtiching focus to main tab
                browser.switch_to.window(windows[0])
                #export data after every product
                #export_data()
                
    except Exception as e_1:
        print(e_1)
        print("Program stoped:")
    export_data()
    print("--- %s seconds ---" % (time.time() - start_time))
    browser.quit()

#defining the main function
def main():
    logging.basicConfig(filename='solution_etsy.log', level=logging.INFO)
    logging.info('Started')
    run_scraper()
    logging.info('Finished')
# Calling the main function 
if __name__ == '__main__':
    main()

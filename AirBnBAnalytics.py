# import module
from PIL import Image
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

df=pd.read_csv('AB_US_2023.csv')
df=df.drop(['neighbourhood_group'],axis=1)

def dataDesc():
    st.header("About our Dataset:")
    status = st.radio("Choose step to perform", ('Dataset structure', 'Dataset columns details'))
     # conditional statement to print 
    # show the result using the success function
    if (status == 'Dataset structure'):
        st.success("Dataset structure")       
        st.write("Cleaned Data set")
        df.tail(10)
        df
    else:
        st.success("Dataset columns details")
        df.info()
        df.columns
def viz1():
    fig1=sns.catplot(x="room_type", y="price", data=df)
    plt.xticks(rotation=45)
    st.pyplot(fig1)        
    
def viz2():
    area_reviews=df.iloc[:6].groupby(['neighbourhood'])['number_of_reviews'].max().reset_index()
    area_reviews=area_reviews.sort_values(by='number_of_reviews',ascending=False)
    fig2= plt.figure(figsize=(6,4))
    area = area_reviews['neighbourhood']
    review =area_reviews['number_of_reviews']
    plt.bar(area, review, color='blue', width=0.6)
    plt.xlabel('neighbourhood')
    plt.ylabel('number of reviews')
    plt.title('Number of reviews by neighbourhood')
    plt.xticks(rotation=45)
    st.pyplot(fig2)

def viz3():
    fig3= plt.figure(figsize=(8,6))
    sns.scatterplot(data=df, x='longitude', y='latitude', hue='neighbourhood', size = 'neighbourhood', palette='deep', sizes=(20, 200), legend ='full')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    plt.title('neighbourhood vs Location')
    st.pyplot(fig3)

def viz4():
    reviews_price = df.groupby(['price'])['number_of_reviews'].max().reset_index()
    reviews_price= reviews_price.head(10)
    price = reviews_price['price']
    review = reviews_price['number_of_reviews']
    fig4 =plt.figure(figsize=(6,4))
    plt.scatter(price, review)
    plt.xlabel('price')
    plt.ylabel('number_of_reviews')
    plt.title('Number of reviews by price')
    st.pyplot(fig4)

def viz5():
    busy_hosts = df.iloc[:6].groupby(['host_name'])['number_of_reviews'].max().reset_index()
    busy_hosts = busy_hosts.sort_values(by='number_of_reviews', ascending=False)
    name_hosts=busy_hosts['host_name']
    review_got=busy_hosts['number_of_reviews']
    fig5=plt.figure(figsize=(8,6))
    plt.bar(name_hosts, review_got, color='blue', width=0.6)
    plt.xlabel('Name of the host')
    plt.ylabel('Review')
    plt.title('Busiest host in terms of reviews')
    st.pyplot(fig5)

def viz6():
    Best_rooms=df.groupby(['room_type'])['minimum_nights'].mean().reset_index()
    Best_rooms=Best_rooms.sort_values(by='minimum_nights',ascending=False).head(10)
    room=Best_rooms['room_type']
    room_stay=Best_rooms['minimum_nights']
    fig6=plt.figure(figsize=(7,5))
    plt.bar(room,room_stay,color='blue',width=0.2)
    plt.xlabel('Room Type')
    plt.ylabel('Minimum Nights Average')
    plt.title('Busiest room type')
    st.pyplot(fig6)

def viz7():
    Best_neighb=df.groupby(['neighbourhood'])['minimum_nights'].mean().reset_index()
    Best_neighb=Best_neighb.sort_values(by='minimum_nights',ascending=False).head(10)
    neigh=Best_neighb['neighbourhood']
    neigh_stay= Best_neighb['minimum_nights']
    fig7=plt.figure(figsize=(7,5))
    plt.bar(neigh,neigh_stay,color='blue',width=0.2)
    plt.xlabel('Neighbourhood')
    plt.ylabel('Minimum Nights Average')
    plt.title('Busiest neighbourhood')
    plt.xticks(rotation=45)
    st.pyplot(fig7)

def viz8():
    
    rt=df.groupby(['room_type'])['room_type'].count().sort_values(ascending=False)
    frt=df['room_type'].unique()
    fig8=plt.figure(figsize=(6,4))
    plt.bar(frt,rt,color='blue',width=0.2)
    plt.xlabel('Room type')
    plt.ylabel('Room type frequency')
    plt.title('Most frequent room type')
    plt.xticks(rotation=45)
    st.pyplot(fig8)



def dataViz():
    st.write("Select Data to visualize:")
    status = st.radio("Choose the data to visualize", ('Price by room type','Reviews Number by neigbourhood', 'Neigbourhood by longitude and latitude','Busiest host in terms of reviews', 'Busiest room type', 'Busiest neighbourhood', 'Most frequent room type'))
    if (status == 'Price by room type'):
        viz1()
    elif (status == 'Reviews Number by neigbourhood'):
        viz2()
    elif (status == 'Neigbourhood by longitude and latitude'):
        viz3()
    #elif (status == 'Number of reviews by price'):
    #    viz4()
    elif (status == 'Busiest host in terms of reviews'):
        viz5()
    elif(status == 'Busiest room type'):
        viz6()
    elif(status == 'Busiest neighbourhood'):
        viz7()
    elif (status == 'Most frequent room type'):
        viz8()
    #elif (status == 'Room number of each type by neigh'):
    #    viz9()

def dataPred():
    x =df[['price']]
    y= df['availability_365']
    
    #Getting Test and Training Set
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.4,random_state=0)
    x_test.head()

    # %%
    lr = LinearRegression()
    lr.fit(x_train,y_train)

    # %%
    
    y_pred = lr.predict(x_test)
    lin_mse = mean_squared_error(y_test, y_pred)
    lin_rmse = np.sqrt(lin_mse)
    st.subheader('Calculated metrics:')
    st.text('Mean Squared Root:')
    st.info(lin_rmse)

    # %%
    r2=r2_score(y_test,y_pred)
    st.text('r2 score')
    st.info(r2) 
    fig=plt.figure(figsize=(7,5))
    plt.scatter(x_test, y_test, color="red")
    plt.plot(x_test,y_pred, color="purple")
    plt.xlabel('Price')
    plt.ylabel('Room Availability')
    plt.title('Availability trend')
    plt.xticks(rotation=45)
    st.pyplot(fig)



# Title
img = Image.open("Logo.png")

# display image using streamlit
# width is used to set the width of an image
st.image(img, width=300)
# st.markdown("<h1>Hello</h1>", unsafe_allow_html=True)

st.title("RBNB Eval !!!")

st.write("Airbnb is an online marketplace for lodging, primarily homestays for vacation rentals,and tourism activities.") 
st.write("In our work, we will evaluate RBNB rooms rental parameters in order to help guests in answering their needs, and hosts to improve their online offers quality.")
st.write("Applying Data Science to Airbnb is challenging. We operate as a two-sided market with significant seasonality and regional variations.")
st.write("Many of these characteristics make it difficult to identify and extract meaningful data from noise within the massive amount of data we get.")
st.write("Since,we would like to share some of our findings with the larger data science community. We offer the chance to examine actual behavioral data and identify its trends in RBNB dataset related to 'Bangkok' famous town.")
# Selection box
 
# first argument takes the titleof the selectionbox
# second argument takes options
Choice = st.selectbox("Please Select one choice",
                     ['Dataset description', 'Data Visualization', 'Data prediction'])
 
if Choice =='Dataset description' :
    dataDesc()
elif Choice =='Data Visualization':
    dataViz()        
else:   
    dataPred()







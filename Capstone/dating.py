#!/usr/local/bin/python3

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB    
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from random import seed, randint    


# Let's get our profiles
all_data = pd.read_csv("profiles.csv")

# Let's list the columns
#for c in sorted(all_data):
#    print(c, end =" ")

# For future reference
# age body_type diet drinks drugs education ethnicity height income job
# last_online location offspring orientation pets religion sex sign smokes
# speaks status
# essay0 essay1 essay2 essay3 essay4 essay5 essay6 essay7 essay8 essay9 

# Let's see how values look like
#for c in sorted(all_data):
#    print("-- " + c + " -----")
#    print(all_data[c].head())

# Let's inspect our data
show = False
if show:
    for c in sorted(all_data):
        # Skip the essays
        if c.startswith("essay"):
            continue 
        print("-- " + c + " -----")
        print(all_data[c].value_counts().sort_index())

# Let's add a sex_code column to our data
r = "sex"
c = "sex_code"
sex_map = {
    "m": 0, 
    "f": 1, 
}
all_data[c] = all_data[r].map(sex_map)

# Let's use the sex column as training labels
# But first remove the NaNs
target_column = "sex_code"
clean_data = all_data.dropna(subset=[target_column])
training_labels = clean_data[target_column]

# Let's use the code provided in the instructions
essay_cols = ["essay0","essay1","essay2","essay3","essay4","essay5","essay6","essay7","essay8","essay9"]
# Removing the NaNs
all_essays = clean_data[essay_cols].replace(np.nan, '', regex=True)
# Combining the essays
all_essays = all_essays[essay_cols].apply(lambda x: ' '.join(x), axis=1)

#for num_row in range(len(all_essays), 10000, -10000):
for num_row in range(10000, 10000, -10000):
    all_essays = all_essays.iloc[0:num_row]
    training_labels = training_labels.iloc[0:num_row]
    counter = CountVectorizer()
    counter.fit(all_essays)
    training_counts = counter.transform(all_essays)
    
    classifier = MultinomialNB()
    classifier.fit(training_counts, training_labels)
    
    # Let's try to classify essays for a few people
    seed(0)
    wrong = 0
    total = 0
    check = 100
    for p in range(check):
        row = randint(1, all_essays.count())
        essay = all_essays[row]
        target = all_data[target_column][row]
        if target != classifier.predict(counter.transform([essay]))[0]:
#            print(target, end=' ')
#            print(classifier.predict(counter.transform([essay])), end=' ')
#            print("-- wrong guess --", end=' ')
#            print(classifier.predict_proba(counter.transform([essay])))
#            print(essay)
            # Inspecting essays for wrong guesses shows a few essays that
            # look more like lists. For the rest, no obvious problem
            wrong += 1
        total += 1
    print(str(wrong) + '/' + str(total) + ' wrong at ' + str(num_row))
    # About 78% correct guesses, so this is significantly better than
    # chance

# Let's add a drinks_code column to our data
r = "drinks"
c = "drinks_code"
drink_map = {
    "not at all": 0, 
    "rarely": 1, 
    "socially": 2, 
    "often": 3, 
    "very often": 4, 
    "desperately": 5,
}
all_data[c] = all_data[r].map(drink_map)

# Let's add a religion_code column to our data
r = "religion"
c = "religion_code"
religion_map = {
    "agnosticism and laughing about it": 0,            
    "agnosticism but not too serious about it": 1,   
    "agnosticism": 2,                                   
    "agnosticism and somewhat serious about it": 3,    
    "agnosticism and very serious about it": 4,        
    "atheism and laughing about it": 0,                
    "atheism but not too serious about it": 1,         
    "atheism": 2,                                       
    "atheism and somewhat serious about it": 3,         
    "atheism and very serious about it": 4,             
    "buddhism and laughing about it": 0,                
    "buddhism but not too serious about it": 1,         
    "buddhism": 2,                                      
    "buddhism and somewhat serious about it": 3,        
    "buddhism and very serious about it": 4,            
    "catholicism and laughing about it": 0,             
    "catholicism but not too serious about it": 1,      
    "catholicism": 2,                                   
    "catholicism and somewhat serious about it": 3,     
    "catholicism and very serious about it": 4,         
    "christianity and laughing about it": 0,            
    "christianity but not too serious about it": 1,     
    "christianity": 2,                                  
    "christianity and somewhat serious about it": 3,    
    "christianity and very serious about it": 4,        
    "hinduism and laughing about it": 0,                
    "hinduism but not too serious about it": 1,         
    "hinduism": 2,                                      
    "hinduism and somewhat serious about it": 3,        
    "hinduism and very serious about it": 4,            
    "islam and laughing about it": 0,                   
    "islam but not too serious about it": 1,            
    "islam": 2,                                         
    "islam and somewhat serious about it": 3,           
    "islam and very serious about it": 4,               
    "judaism and laughing about it": 0,                 
    "judaism but not too serious about it": 1,          
    "judaism": 2,                                       
    "judaism and somewhat serious about it": 3,         
    "judaism and very serious about it": 4,             
    "other and laughing about it": 0,                   
    "other but not too serious about it": 1,            
    "other": 2,                                         
    "other and somewhat serious about it": 3,           
    "other and very serious about it": 4,               
}
all_data[c] = all_data[r].map(religion_map)

# Let's plot income vs age in a scatterplot
# This is where our plots in the slides comes from 
# Then try to predict income from various factors
show = True
if show:
    # Clean the data
    clean_data = all_data.dropna(subset=["age", "height", "income", "religion_code", "sex_code"])
    clean_data = clean_data[clean_data["income"] > -1]
    clean_data = clean_data[clean_data["income"] <= 100000]
    # Show a scatter plot of age, height and sex
    sc = plt.scatter(clean_data[["age"]], clean_data[["height"]], c=clean_data[["sex_code"]], cmap="Set3", alpha=0.1)
    plt.xlabel("age")
    plt.xlim(15, 72)
    plt.ylabel("height")
    plt.ylim(55, 85)
    for sex in ['m', 'f']:
        plt.scatter([], [], color=sc.get_cmap()(sc.norm(sex_map[sex])), alpha=0.3, label=sex)
    plt.legend(scatterpoints=1, frameon=False, labelspacing=1, title='Color')
    plt.legend()
    plt.show()
    plt.close()
    # Show a scatter plot of age and income
    plt.scatter(clean_data[["age"]], clean_data[["income"]], alpha=0.1)
    plt.xlabel("age")
    plt.xlim(15, 72)
    plt.ylabel("income")
    plt.show()
    plt.close()
    # Scale the data
    feature_data = clean_data[["age", "height", "religion_code", "sex_code"]]
    x = feature_data.values
    min_max_scaler = MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    feature_data = pd.DataFrame(x_scaled, columns=feature_data.columns)
    # Now we run the regression
    y = clean_data[["income"]]
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size = 0.2, random_state=6)
    lm = LinearRegression()
    model = lm.fit(x_train, y_train)
    y_predict = lm.predict(x_test)
    # Score train and test results
    print("Train score:")
    print(lm.score(x_train, y_train))
    print("Test score:")
    print(lm.score(x_test, y_test))

## That's all!

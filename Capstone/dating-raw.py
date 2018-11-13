#!/usr/local/bin/python3

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

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
        if c.startswith("essay"):
            continue 
        print("-- " + c + " -----")
        print(all_data[c].value_counts().sort_index())
        # Wait for enter before moving to next column
#        try:
#            input("Press [enter] to continue")
#        except SyntaxError:
#            pass

# Let's plot the age distribution
show = False
c = "age"
if show:
    print(all_data[c].value_counts().sort_index())
    plt.hist(all_data[c], bins=2+110-19)
    plt.xlabel(c)
    plt.ylabel("Frequency")
    plt.xlim(18, 70)
    plt.show()
    plt.close()

# Let's map and plot body_type frequency
show = False
r = "body_type"
c = "body_type_code"
if show:
    print(all_data[r].value_counts().sort_index())
    body_type_map = {
        "rather not say": 0,
        "skinny": 1,
        "thin": 2,
        "athletic": 3,
        "average": 4,
        "fit": 5,
        "jacked": 6,
        "full figured": 7,
        "a little extra": 8,
        "curvy": 9,
        "overweight": 10,
        "used up": 11,
    }
    all_data[c] = all_data[r].map(body_type_map)
    print(all_data[c].value_counts().sort_index())
    plt.hist(all_data[c], bins=12)
    plt.xlabel(c)
    plt.ylabel("Frequency")
    plt.show()
    plt.close()
    # Not exactly a nice bell curve. Maybe try to split weight and fitness?

# Let's map and plot the diet distribution
show = False
r = "diet"
c = "diet_code"
if show:
    print(all_data[r].value_counts().sort_index())
    diet_map_1 = {
        "anything":            "anything",  
        "halal":               "halal",	    
        "kosher":              "kosher",    
        "other":               "other",	    
        "vegan":               "vegan",	    
        "vegetarian":          "vegetarian",
        "mostly anything":     "anything",  
        "mostly halal":        "halal",	    
        "mostly kosher":       "kosher",    
        "mostly other":        "other",	    
        "mostly vegan":        "vegan",	    
        "mostly vegetarian":   "vegetarian",
        "strictly anything":   "anything",  
        "strictly halal":      "halal",	    
        "strictly kosher":     "kosher",    
        "strictly other":      "other",	    
        "strictly vegan":      "vegan",	    
        "strictly vegetarian": "vegetarian",
    }
    diet_map_2 = {
        "mostly anything":     0,
        "mostly halal":        0,
        "mostly kosher":       0,
        "mostly other":        0,
        "mostly vegan":        0,
        "mostly vegetarian":   0,
        "anything":            1,
        "halal":               1,
        "kosher":              1,
        "other":               1,
        "vegan":               1,
        "vegetarian":          1,
        "strictly anything":   2,
        "strictly halal":      2,
        "strictly kosher":     2,
        "strictly other":      2,
        "strictly vegan":      2,
        "strictly vegetarian": 2,
    }
    all_data[c] = all_data[r].map(diet_map_2)
    print(all_data[c].value_counts().sort_index())
    plt.hist(all_data[c], bins=3)
    plt.xlabel(c)
    plt.ylabel("Frequency")
    plt.show()
    plt.close()

# Let's plot the income distribution
show = False
c = "income"
if show:
    print(all_data[c].value_counts().sort_index())
    plt.hist(all_data[c], bins=2+100-2)
    plt.xlabel(c)
    plt.ylabel("Frequency")
    plt.xlim(20000, 1000000)
    plt.ylim(0, 3000)
    plt.show()
    plt.close()
    # We observe gaps in the graph, we assume that the questions asked
    # were in the form: do you earn more than xxx? We might need to
    # smoothen things if we want to use this data. On another hand, many
    # people (about 80%) don't respond, so this doesn't look too promising
    # by itself. Still might be useful combined with other columns

# Let's see what we can get some useful classification from the essays:
# - Format the data as in Naive Bayes Classifier lesson, part 7
# - We need to target one or more other columns, maybe job or age
#   (brackets)?
# - Let's try ages, as we can start simple (young vs old) then refine if
# promising 

from sklearn.feature_extraction.text import CountVectorizer

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
#clean_data = all_data.dropna(subset=["sex"])
#training_labels = clean_data["sex"]
target_column = "sex_code"
clean_data = all_data.dropna(subset=[target_column])
training_labels = clean_data[target_column]

# Let's use the code provided in the instructions
essay_cols = ["essay0","essay1","essay2","essay3","essay4","essay5","essay6","essay7","essay8","essay9"]
#essay_cols = ["body_type", "diet", "drinks", "drugs", "ethnicity", "job", "location", "offspring", "orientation", "pets", "religion", "sex", "sign", "smokes", "speaks", "status"]
# Removing the NaNs
all_essays = clean_data[essay_cols].replace(np.nan, '', regex=True)
# Combining the essays
all_essays = all_essays[essay_cols].apply(lambda x: ' '.join(x), axis=1)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB    
from random import seed, randint    

for num_row in range(len(all_essays), 1000, -1000):
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
            print(target, end=' ')
            print(classifier.predict(counter.transform([essay])), end=' ')
            # Does orientation look significant for mistakes
#           print(all_data["orientation"][row], end=' ')
            # 4/18 identify as gay, 1/18 as bisexual
#           print("-- wrong guess --", end=' ')
            print(classifier.predict_proba(counter.transform([essay])))
#           print(essay)
            # Inspecting essays for wrong guesses shows a few essays that are
            # structured as lists. For the rest, no obvious problem
            wrong += 1
        total += 1
    print(str(wrong) + '/' + str(total) + ' wrong at ' + str(num_row))
    # About 78% correct guesses, so this is significantly better than chance
# We could remove infrequent words (typos, etc) and see if that helps

# Let's use the age column as training labels
training_labels = all_data["age"]
classifier = MultinomialNB()
#classifier.fit(training_counts, training_labels)

# Let's try again to classify essays for a few people
from random import seed, randint

seed(0)
wrong = 0
total = 0
check = 0
for p in range(check):
    row = randint(0, all_essays.count())
    essay = all_essays[row]
    age = all_data["age"][row]
    diff = age - classifier.predict(counter.transform([essay])[0])
    if diff > 5 or diff < -5:
        print(diff, end=' ')
        print("-- wrong guess --", end=' ')
        print(classifier.predict_proba(counter.transform([essay])))
#        print(essay)
        wrong += 1
    total += 1
print(str(wrong) + '/' + str(total) + ' wrong')

# Let's see what we find about industry vs education (okay, maybe later)

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

# Let's see what we find about religion vs drinking
# Let's map and plot drinking
show = False
if show:
    print(all_data[r].value_counts().sort_index())
    print(all_data[c].value_counts().sort_index())
    plt.hist(all_data[c], bins=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
    plt.xlabel(c)
    plt.ylabel("Frequency")
    plt.show()
    plt.close()

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

# Let's plot religious intensity
show = False
if show:
    print(all_data[r].value_counts().sort_index())
    print(all_data[c].value_counts().sort_index())
    plt.hist(all_data[c], bins=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5])
    plt.xlabel(c)
    plt.ylabel("Frequency")
    plt.show()
    plt.close()

# Let's add a religion_color column to our data
r = "religion"
c = "religion_color"
religion_map = {
    "agnosticism and laughing about it": 0,
    "agnosticism but not too serious about it": 0,
    "agnosticism": 0,
    "agnosticism and somewhat serious about it": 0,
    "agnosticism and very serious about it": 0,
    "atheism and laughing about it": 1,
    "atheism but not too serious about it": 1,
    "atheism": 1,
    "atheism and somewhat serious about it": 1,
    "atheism and very serious about it": 1,
    "buddhism and laughing about it": 2,
    "buddhism but not too serious about it": 2,
    "buddhism": 2,
    "buddhism and somewhat serious about it": 2,
    "buddhism and very serious about it": 2,
    "catholicism and laughing about it": 3,
    "catholicism but not too serious about it": 3,
    "catholicism": 3,
    "catholicism and somewhat serious about it": 3,
    "catholicism and very serious about it": 3,
    "christianity and laughing about it": 4,
    "christianity but not too serious about it": 4,
    "christianity": 4,
    "christianity and somewhat serious about it": 4,
    "christianity and very serious about it": 4,
    "hinduism and laughing about it": 5,
    "hinduism but not too serious about it": 5,
    "hinduism": 5,
    "hinduism and somewhat serious about it": 5,
    "hinduism and very serious about it": 5,
    "islam and laughing about it": 6,
    "islam but not too serious about it": 6,
    "islam": 6,
    "islam and somewhat serious about it": 6,
    "islam and very serious about it": 6,
    "judaism and laughing about it": 7,
    "judaism but not too serious about it": 7,
    "judaism": 7,
    "judaism and somewhat serious about it": 7,
    "judaism and very serious about it": 7,
    "other and laughing about it": 8,
    "other but not too serious about it": 8,
    "other": 8,
    "other and somewhat serious about it": 8,
    "other and very serious about it": 8,
}
all_data[c] = all_data[r].map(religion_map)

# Let's map and plot religion
show = False
if show:
    print(all_data["drinks"].value_counts())
    print(all_data["drinks_code"].value_counts().sort_index())
    print(all_data["religion"].value_counts())
    print(all_data["religion_color"].value_counts().sort_index())
    print(all_data["religion_code"].value_counts().sort_index())
#    plt.hist(all_data["religion_color"], bins=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5])
#    plt.xlabel("Religion")
#    plt.xlabel("Frequency")
#    plt.show()
#    plt.close()

    # Let's summarise religion
    religion = all_data.groupby(["religion"]).count()
    print(type(religion))

##    for c in sorted(religion):
##        if c.startswith("essay"):
##            continue 
##        print("-- " + c + " -----")
##        print(all_data[c].value_counts().sort_index())
    
#    plt.scatter(all_data[["drinks_code"]], all_data[["religion_color"]], c=all_data[["religion_color"]], cmap="Set1", alpha=0.1)
#    plt.xlabel("Religion")
#    plt.xlim(-0.5, 8.5)
#    plt.ylabel("Attitude")
#    plt.ylim(-0.5, 4.5)
#    plt.show()
#    plt.close()

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Let's plot age vs height in a scatterplot
show = False
if show:
    # Clean the data
    clean_data = all_data.dropna(subset=["age", "height", "sex_code", "income"])
    clean_data = clean_data[clean_data["income"] != -1]
    # Show a scatter plot
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
    # Scale the data
    feature_data = clean_data[["age", "height", "sex_code"]]
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

# Let's plot income vs age in a scatterplot
show = False
if show:
    # Clean the data
    clean_data = all_data.dropna(subset=["age", "height", "income", "religion_code", "sex_code"])
    clean_data = clean_data[clean_data["income"] > -1]
    clean_data = clean_data[clean_data["income"] <= 100000]
    # Show a scatter plot
    plt.scatter(clean_data[["age"]], clean_data[["income"]], alpha=0.1)
    plt.xlabel("age")
    plt.xlim(15, 72)
    plt.ylabel("income")
#    plt.ylim(55, 85)
#    plt.show()
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

# Drinks and Religion code need to be set above
show = False
if show:
    # Let's remove NaNs and split the set
#    clean_data = all_data.dropna(subset=["age", "sex_code", "height"])
#    x = clean_data[["age", "sex_code"]]
#    y = clean_data[["height"]]
    clean_data = all_data.dropna(subset=["religion_code", "drinks_code"])
    x = clean_data[["religion_code"]]
    y = clean_data[["drinks_code"]]
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size = 0.2, random_state=6)
    lm = LinearRegression()
    model = lm.fit(x_train, y_train)
    y_predict = lm.predict(x_test)

    print("Train score:")
    print(lm.score(x_train, y_train))
    print("Test score:")
    print(lm.score(x_test, y_test))

    plt.scatter(y_test, y_predict)
    plt.xlabel("Actual Drinks")
    plt.ylabel("Predicted Drinks")
    plt.title("Actual Drinks vs Predicted Drinks")
    plt.show()
    plt.close()
    
#===

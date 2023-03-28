from firebase_admin import credentials

import firebase_admin
from flask import Flask, request, jsonify
import json
from firebase_admin import credentials
from firebase_admin import firestore
from firebase_admin import initialize_app
import firebase_admin
import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
# from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import re
import json
import requests
app = Flask(__name__)


@app.route('/', methods=['POST'])
def getSimilarity():

    global response
    if (request.method == "POST"):
        request_data = request.data
        request_data = json.loads(request_data.decode('utf-8'))
        if not firebase_admin._apps:
            # reed the key og firebase from  ewaa apllication
            cred = credentials.Certificate(
                request_data['cer'])
            firebase_admin.initialize_app(cred)

        db = firestore.client()
        # Get all pet from firebase
        pets = list(db.collection('pets').stream())
        most_like = []
        # Get the most ten like pet
        mostlike = db.collection('pets').where("isAdopted", "==", False).order_by(
            "likes_count", 'DESCENDING').limit(10).get()
        for i in mostlike:
            print(i.to_dict())
            j = i.to_dict()
            most_like.append(j['petId'])
        print(most_like)
        least_like = []
        leastlike = db.collection('pets').where("isAdopted", "==", False).order_by("likes_count", 'ASCENDING').limit(
            10).get()
        for i in leastlike:
            print(i.to_dict())
            j = i.to_dict()
            least_like.append(j['petId'])
        print(least_like)
        # tranform the stram to dic and save it to datafram
        pets_dict = list(map(lambda x: x.to_dict(), pets))
        df = pd.DataFrame(pets_dict)
        # get the user id of the active user
        print(request_data['userID'])
        df.info()
        # initialize dataframe to save each pet and who like this pet
        data2 = {"user": [], "petId": []}
        df2 = pd.DataFrame(data2)
        df2

        # get like user for eavh pet and save it ti the previous datafram
        for index, row in df.iterrows():
            petid = row['petId']
            for j in row['likedUsers']:
                print(j)
                print(petid)
                new_row = {"user": j, "petId": petid}
                # append row to the dataframe
                df2 = df2.append(new_row, ignore_index=True)

        df_cd = pd.merge(df, df2, how='inner', on='petId')
        print(df2)
       # get like list for this user
        df_cd = pd.merge(df, df2, how='inner', on='petId')
        # get the like pet for active user
        likeList = df_cd.loc[df_cd['user'] == request_data['userID']]
        print(likeList.empty)
        # if the likelist is empty then we return the most like pet for content based and for collabrative we retuen the least like pet
        if (likeList.empty):
            return {"similarity_pets": most_like, "similarity_users": least_like}

        import numpy as np

        copy = likeList.copy(deep=True)
        # genrate crosee tab to encode the color for each pet
        color = pd.crosstab(df['petId'], df['color'])

        color

        likecolor = color[color.index.isin(np.array(likeList.petId))]
        print(likecolor)
        print(likecolor.mean())

        breed = pd.crosstab(df['petId'], df['breed'])
        print(breed)
        likebreed = breed[breed.index.isin(np.array(likeList.petId))]
        print(likebreed)
        print(likebreed.mean())

        age = pd.crosstab(df['petId'], df['age'])
        pd.crosstab(df['petId'], df['age'])
        likeage = age[age.index.isin(np.array(likeList.petId))]
        print(likeage)
        print(likeage.mean())

        Category = pd.crosstab(df['petId'], df['category'])
        pd.crosstab(df['petId'], df['category'])
        likeCategory = Category[Category.index.isin(np.array(likeList.petId))]
        likeCategory
        likeCategory.mean()

        Weight = []
        # Weight.append(1)
        for i in likecolor.mean().values:
            Weight.append(i)
        for i in likebreed.mean().values:
            Weight.append(i)
        for i in likeage.mean().values:
            Weight.append(i)
        for i in likeCategory.mean().values:
            Weight.append(i)

        from functools import reduce

        # define list of DataFrames
        dfs = [pd.DataFrame(color), pd.DataFrame(
            breed), pd.DataFrame(age), pd.DataFrame(Category)]

        # merge all DataFrames into one
        final_df = reduce(lambda left, right: pd.merge(left, right, on=['petId'],
                                                       how='outer'), dfs)
        list_user_like = likeList['petId']
        final_df = final_df.drop(list_user_like, axis=0)

        r = final_df.multiply(Weight, axis=1)
        r
        r['Total'] = r.sum(axis=1)
        r.sort_values(by=['Total'], ascending=False).head(20)
        isAdopted_Like = likeList[likeList['isAdopted'] == True]
        isAdopted_Like = isAdopted_Like['petId']
        isAdopted_Like
        df5 = df[~df['petId'].isin(isAdopted_Like)]
        df5 = df[~df['petId'].isin(likeList.petId)]
        df5 = df5[df5['isAdopted'] == True]

        petid_adopted = df5['petId']
        print(petid_adopted)
        final_reco = r.drop(np.array(petid_adopted))
        # get id of my pey
        myPet = db.collection('pets').where(
            "ownerId", "==", request_data['userID']).get()
        my_pet = []
        for i in myPet:
            print(i.to_dict())
            j = i.to_dict()
            my_pet.append(j['petId'])
       # delete my pet
        final_reco = final_reco[~final_reco.index.isin(my_pet)]

        recomm = final_reco.sort_values(by=['Total'], ascending=False).head(10)
        print(recomm['Total'])
        recommandation = []
        for index, row in recomm.iterrows():
            print(index)
            if (row['Total'] > 0):
                recommandation.append(index)
        if (len(recommandation) == 0):
            print(f' content recommander is empty? {True}')
            recommandation = most_like

        print(recommandation)

        ###### Collabrative #####################
        # We build an n X m matrix consisting of the likes of n users and m pets.
        user_item_coll = pd.crosstab(df2.user, df2.petId)

        # we use cosine similrity to find the similrity between users
        users_similarity_cosine = pd.DataFrame(cosine_similarity(user_item_coll),
                                               index=user_item_coll.index, columns=user_item_coll.index)
        users_similarity_cosine
        users_similarity_cosine.drop(
            index=request_data['userID'], inplace=True)
        users_similarity_cosine

        # we ues the threhold  to find  similarity between the users and decide whether two users are similar or not
        threshold = 0.5
        # specify the maxmium number of similar users since the recommended pet at must be 10
        n = 10
        # then we picked the most simiries users based on the threshold
        most_similar_user = \
            users_similarity_cosine[users_similarity_cosine[request_data['userID']] > threshold][
                request_data['userID']].sort_values(ascending=False)[:n]
        print(most_similar_user.empty)
        if most_similar_user.empty:
            print(True)
            return {"similarity_pets": recommandation, "similarity_users":  least_like}

        # drop the pets that user was liked
        user_item_for_most_sim_users = user_item_coll.drop(
            np.array(list_user_like), axis=1)
        user_item_for_most_sim_users
        print(user_item_for_most_sim_users)
        # drop the users that is not similar with active user
        user_item_for_most_sim_users = user_item_for_most_sim_users[
            user_item_for_most_sim_users.index.isin(most_similar_user.index)]
        user_item_for_most_sim_users
        print(user_item_for_most_sim_users)
        # Pets that are not liked by similar users are dropped
        return_like_pet_for_all_most_sim_users = user_item_for_most_sim_users.drop(
            columns=user_item_for_most_sim_users.columns[user_item_for_most_sim_users.sum() == 0])
        #
        if (return_like_pet_for_all_most_sim_users.empty):
            print(True)
        item_score = {}
        item_like_count = {}
        a = []
        # For each pet, find the number of users who like it
        for i in return_like_pet_for_all_most_sim_users.columns:
            count = 0
            col = return_like_pet_for_all_most_sim_users[i]
            count = sum(col.values)
            item_like_count[i] = count
        # muitiplay each row for each user by the cosine similarity between this user and the active user to calculate weightd matrix
        for index, row in return_like_pet_for_all_most_sim_users.iterrows():
            print(row * most_similar_user[index])
            y = row * most_similar_user[index]
            #  print(y.values)
            a.append(y.values)

        # weightd matrix
        c = pd.DataFrame(a, index=return_like_pet_for_all_most_sim_users.index,
                         columns=return_like_pet_for_all_most_sim_users.columns)

        for co in c.columns:
            if (co != ""'total'):
                print(c[co].sum())
                item_score[co] = c[co].sum()

       # normailze weightd matrix
        total_score_norm = []
        for j in item_score:
            total_score_norm.append(item_score[j] / item_like_count[j])

        print(item_score)
        print(item_like_count)
        rec2 = pd.DataFrame(
            total_score_norm, index=return_like_pet_for_all_most_sim_users.columns, columns=['score'])

        rec2 = rec2.sort_values(by=['score'], ascending=False)[:n]
        print(rec2)
        df6 = df[df['isAdopted'] == True]
        petid_adopted2 = df6['petId']
        rec2 = rec2[~rec2.index.isin(petid_adopted2)]
        rec2 = rec2[~rec2.index.isin(my_pet)]

        Collaborative_filtering = []
        for index, row in rec2.iterrows():
            print(index)
            if (row['score'] > 0):
                Collaborative_filtering.append(index)

        if (len(Collaborative_filtering) == 0):
            print(f' collbrative recommander is empty? {True}')
            Collaborative_filtering = least_like

    return {"similarity_pets": recommandation, "similarity_users": Collaborative_filtering}


if __name__ == "__main__":
    app.run(host="0.0.0.0")

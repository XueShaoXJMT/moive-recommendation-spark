import pandas as pd

# define a function to package the recommendation
def Recommend(df, model, k,id):
    '''
    k: the number of movies to recommend
    id: the id of the user to give recommendations
    model: the trained model for recommendation
    '''
    # the table for all top10 recommendations
    all=model.recommendForAllUsers(k)
    user=all.where(all.userId==id).toPandas()
    if user.shape[0]==0:
        print('No user with id '+str(id)+' is found in the data.')
        return None
    user=user.iloc[0,1]
    user=pd.DataFrame(user,columns=['movieId','predicted_ratings'])
    temp=None
    for i in user['movieId']:
        if not tmp:
            tmp=df.where(df.movieId==str(i))
        else:
            tmp=tmp.union(df.where(df.movieId==str(i)))
    out=pd.concat([tmp.toPandas(),user['predicted_ratings']],axis=1)
    out.index=range(1,k+1)
    return out


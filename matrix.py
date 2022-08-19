# method 1: Euclidean distance based similarity
# the smaller the euclidean distance between the factors, the similar the movies
# this similarity considers the actual strength,
# e.g. movie 1 with factor [1,2,3] and movie 2 with factor [2,4,6] are considered not similar enough
from main import spark

def dist_sim(df,model,k,mid):
    '''
    k: number of similar movies to find
    mid: id of the movie to find similarities
    '''
    info=spark.sql('select * from movie_factors where movieId='+str(mid)).toPandas()
    if info.shape[0]<=0:
        print('No movie with id '+str(mid)+'.')
        return None, None
    tmp=['select movieId,']
    for i in range(model.rank):
        val=info.iloc[0,i+1]
        if val>0:
            cmd='feature'+str(i)+'-'+str(val)
        else:
            cmd='feature'+str(i)+'+'+str(-val)
        if i<model.rank-1:
            tmp.append('('+cmd+')*('+cmd+') as sd'+str(i)+',')
        else:
            tmp.append('('+cmd+')*('+cmd+') as sd'+str(i))
    tmp.append('from movie_factors where movieId!='+str(mid))
    ssd=spark.sql(' '.join(tmp))
    ssd=ssd.selectExpr('movieId','sd0+sd1+sd2+sd2+sd4 as ssd').orderBy('ssd').limit(k).toPandas()
    out=None
    for i in ssd['movieId']:
        if not out:
            out=df.where(df.movieId==str(i))
        else:
            out=out.union(df.where(df.movieId==str(i)))
    out=out.toPandas()
    out.index=range(1,k+1)
    return out, ssd

# method 2: cosine similarity
# the larger the cosine value, the smaller the two feature vectors' angle, the similar the movies
# this similarity considers the direction only,
# e.g. movie 1 with factor [1,2,3] and movie 2 with factor [2,4,6] are considered the same
def cos_sim(df,model,k,mid):
    '''
    k: number of similar movies to find
    mid: id of the movie to find similarities
    '''
    info=spark.sql('select * from movie_factors where movieId='+str(mid)).toPandas()
    if info.shape[0]<=0:
        print('No movie with id '+str(mid)+' is found in the data.')
        return None, None
    norm_m=sum(info.iloc[0,1:].values**2)**0.5
    tmp=['select movieId,']
    norm_str=['sqrt(']
    for i in range(model.rank):
        cmd='feature'+str(i)+'*'+str(info.iloc[0,i+1])
        tmp.append(cmd+' as inner'+str(i)+',')
        if i<model.rank-1:
            norm_str.append('feature'+str(i)+'*feature'+str(i)+'+')
        else:
            norm_str.append('feature'+str(i)+'*feature'+str(i))
    norm_str.append(') as norm')
    tmp.append(''.join(norm_str))
    tmp.append(' from movie_factors where movieId!='+str(mid))
    inner=spark.sql(' '.join(tmp))
    inner=inner.selectExpr('movieId',\
                         '(inner0+inner1+inner2+inner3+inner4)/norm/'+str(norm_m)+' as innerP').\
                         orderBy('innerP',ascending=False).limit(k).toPandas()
    out=None
    for i in inner['movieId']:
        if not out:
            out=df.where(df.movieId==str(i))
        else:
            out=out.union(df.where(df.movieId==str(i)))
    out=out.toPandas()
    out.index=range(1,k+1)
    return out, inner

def show_matrix(model):
    # access the movie factor matrix
    factors=model.itemFactors
    factors.printSchema()
    comd=["movie_factors.selectExpr('id as movieId',"]
    for i in range(model.rank):
        if i<model.rank-1:
            comd.append("'features["+str(i)+"] as feature"+str(i)+"',")
        else:
            comd.append("'features["+str(i)+"] as feature"+str(i)+"'")
    comd.append(')')
    movie_factors=eval(''.join(comd))
    movie_factors.createOrReplaceTempView('movie_factors')
    movie_factors.show()




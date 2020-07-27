import datetime
import src.settings.config as config

def exp__level(exp):
    if 0 <= exp < 5:
        return 'junior'
    if 5 <= exp < 15:
        return 'senior'
    return 'expert'

def profil(age):
    if age >= 47:
        return 'BP'
    return 'GP'

def create_date_features(df, column):
    df[column] = df[column].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'))
    df['day_{}'.format(column)] = df[column].dt.day
    df['week_{}'.format(column)] = df[column].dt.week
    df['month_{}'.format(column)] = df[column].dt.month
    df['weekday_{}'.format(column)] = df[column].dt.weekday
    df = df.drop(column, axis=1)
    return df
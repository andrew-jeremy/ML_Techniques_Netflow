
import ipaddress

def replace_boolean(data):
    for col in data:
        data[col].replace(True, 1, inplace=True)
        data[col].replace(False, 0, inplace=True)
    return data

def convert_to_int(ip:str):
    try:
        return int(ipaddress.ip_address(ip))
    except:
        return 0
               
def feat_eng(df_out):
    # features to drop
    l1 = ['flowsrcname','site','flowtype','inputname','tags','action','customer','dstas.org',
            'dstiprep.categories', 'dstgeo.continentcode', 'dstgeo.countrycode',
            'dstgeo.subdiso', 'srcas.org', 'srciprep.categories',
            'srcgeo.continentcode', 'srcgeo.countrycode', 'srcgeo.subdiso', 'dstiprep.count','tos',
            'bogondst','bogonsrc','dstinternal','input','output','srciprep.count']
            
    l2 = ['dstvlan','inputalias','inputclasses','outputalias','outputclasses',
        'outputname','payload','dstowneras.org','srcvlan','dstowneras.number','srcas.number','srcowneras.number','srcowneras.org','dstas.number']

    l3 = ['flowrtime','end','flowversion','ipversion']

    #tcp = [col for col in df_out.columns if 'tcp' in col]
    icm = [col for col in df_out.columns if 'icm' in col]

    #l = l1 + l2 + l3 + tcp + icm
    l = l1 + l2 + l3  + icm
    df_out = df_out.drop(l,axis=1)
    df_out = replace_boolean(df_out)
    df_out[['dstip', 'flowsrcip', 'nexthop', 'srcip']] = df_out[['dstip', 'flowsrcip', 'nexthop', 'srcip']].fillna('0.0.0.0')
    df_out[['dstip', 'flowsrcip', 'nexthop', 'srcip']] = df_out[['dstip', 'flowsrcip', 'nexthop', 'srcip']].astype(str)
    
    # convert to string type for later hotencoding
    # ['srcport','dstport','protocolint','dstgeo.location.lat',
    # 'dstgeo.location.lon','srcgeo.location.lat','srcgeo.location.lon'] 
    df_out[['srcport','dstport','protocolint','dstgeo.location.lat','dstgeo.location.lon','srcgeo.location.lat','srcgeo.location.lon']]\
    = df_out[['srcport','dstport','protocolint','dstgeo.location.lat','dstgeo.location.lon','srcgeo.location.lat','srcgeo.location.lon']].astype(str)
    df_out['timeDelta'] = df_out["timestamp"] - df_out["start"]
    df_out.drop(["timestamp","start"], axis=1, inplace=True)

    #df_out['transferred_ratio'] = (df_out['bits'] / 8 / df_out['packets']) / (df['duration'] + 1)
    #df_out.drop(columns = ['duration', 'packets', 'bits'], axis =1, inplace=True)

    return df_out.to_dict('records')[0]  # in form to be used by online river autoencoder anomaly detector

def feat_select(df_out):
    # features to drop
    l1 = ['flowsrcname','site','flowtype','inputname','tags','action','customer','dstas.org',
            'dstiprep.categories', 'dstgeo.continentcode', 'dstgeo.countrycode',
            'dstgeo.subdiso', 'srcas.org', 'srciprep.categories',
            'srcgeo.continentcode', 'srcgeo.countrycode', 'srcgeo.subdiso', 'dstiprep.count','tos',
            'bogondst','bogonsrc','dstinternal','input','output','srciprep.count']
            
    l2 = ['dstvlan','inputalias','inputclasses','outputalias','outputclasses',
        'outputname','payload','dstowneras.org','srcvlan','dstowneras.number','srcas.number','srcowneras.number','srcowneras.org','dstas.number']

    l3 = ['flowrtime','end','flowversion','ipversion']

    #tcp = [col for col in df_out.columns if 'tcp' in col]
    icm = [col for col in df_out.columns if 'icm' in col]

    #l = l1 + l2 + l3 + tcp + icm
    l = l1 + l2 + l3  + icm
    df_out = df_out.drop(l,axis=1)
    df_out = replace_boolean(df_out)
    df_out[['dstip', 'flowsrcip', 'nexthop', 'srcip']] = df_out[['dstip', 'flowsrcip', 'nexthop', 'srcip']].fillna('0.0.0.0')
    df_out[['dstip', 'flowsrcip', 'nexthop', 'srcip']] = df_out[['dstip', 'flowsrcip', 'nexthop', 'srcip']].astype(str)

    # convert to string type for later hotencoding
    # ['srcport','dstport','protocolint','dstgeo.location.lat',
    # 'dstgeo.location.lon','srcgeo.location.lat','srcgeo.location.lon'] 
    df_out[['srcport','dstport','protocolint','dstgeo.location.lat','dstgeo.location.lon','srcgeo.location.lat','srcgeo.location.lon']]\
    = df_out[['srcport','dstport','protocolint','dstgeo.location.lat','dstgeo.location.lon','srcgeo.location.lat','srcgeo.location.lon']].astype(str)
    df_out['timeDelta'] = df_out["timestamp"] - df_out["start"]
    df_out.drop(["timestamp","start"], axis=1, inplace=True)

    #df_out['transferred_ratio'] = (df_out['bits'] / 8 / df_out['packets']) / (df['duration'] + 1)
    #df_out.drop(columns = ['duration', 'packets', 'bits'], axis =1, inplace=True)
    df_num = df_out.select_dtypes(exclude=['object'])
    df_cat = df_out.select_dtypes(include=['object'])
    df_cat[['dstip', 'flowsrcip', 'nexthop', 'srcip']] = df_cat[['dstip', 'flowsrcip', 'nexthop', 'srcip']].applymap(lambda x: convert_to_int(x))
    df_num[['dstgeo.location.lat','dstgeo.location.lon','srcgeo.location.lat','srcgeo.location.lon']] =\
    df_cat[['dstgeo.location.lat','dstgeo.location.lon','srcgeo.location.lat','srcgeo.location.lon']].astype(float)
    df_cat.drop(['dstgeo.location.lat','dstgeo.location.lon','srcgeo.location.lat','srcgeo.location.lon'], axis=1, inplace=True)
    df_cat = df_cat.astype(int)
    return df_num.to_dict('records')[0],df_cat.to_dict('records')[0]  # in form to be used by online river autoencoder anomaly detector
# coding: utf8
import preprocessing
import numpy as np

def checkStationsStaticValues(data):
    station_datas = preprocessing.separateStationDatas(data)
    static_keys = ['hlres_50', 'green_5000', 'hldres_50', 'route_100', 'hlres_1000', 'route_1000', 'roadinvdist', 'port_5000', 'hldres_100', 'natural_5000', 'hlres_300', 'hldres_300', 'route_300', 'route_500', 'hlres_500', 'hlres_100', 'industry_1000', 'hldres_500', 'hldres_1000']

    for station_id_key in station_datas:
        print 'checking static values of station ', station_id_key
        station_data = station_datas[station_id_key]

        for i, key in enumerate(station_data.columns):
            if key in static_keys and len(set(station_data[key])) > 1:
                print '->Problem with station ', station_id_key, ' static key ',key, ' has values ',set(station_data[key])
            elif key in static_keys:
                print '->key ', key, ' ok.'

def checkEqualityOfDynamicValuesInZone(data):
    keys_to_drop = [
        'hlres_50', 'green_5000', 'hldres_50', 'route_100', 'hlres_1000',
        'route_1000', 'roadinvdist', 'port_5000', 'hldres_100', 'natural_5000',
        'hlres_300', 'hldres_300', 'route_300', 'route_500', 'hlres_500', 'hlres_100',
        'industry_1000',  'hldres_500', 'hldres_1000', 'ID', 'pollutant', 'y'
    ]
    data = data.drop(keys_to_drop, axis=1)

    zone_ids = [0, 1, 2, 3, 4, 5]

    dynamic_keys = [
        'temperature', 'precipprobability', 'precipintensity',
        'windbearingcos','windbearingsin', 'windspeed','cloudcover',
        'pressure'
    ]

    n_equality = 0
    n_inequality = 0

    for zone_id in zone_ids:
        zone_data = data[data['zone_id'] == zone_id]
        daytime_values = np.unique(zone_data['daytime'].values)

        for daytime_value in daytime_values:
            zone_daytime_data = zone_data[zone_data['daytime'] == daytime_value]

            isEqual = True
            for dynamic_key in dynamic_keys:
                if len(np.unique(zone_daytime_data[dynamic_key].values)) > 1:
                    isEqual = False
                    break

            if isEqual:
                n_equality += 1
            else:
                n_inequality += 1

    print 'equality cases : ', n_equality, ' ; inequality case : ', n_inequality

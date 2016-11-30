# coding: utf8
import preprocessing

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

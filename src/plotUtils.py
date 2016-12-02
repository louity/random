import matplotlib.pyplot as plt

def plot_histogram(values, pollutant):
    plt.figure(1);
    plt.hist(values, bins=50);
    plt.title(pollutant + ' values histogram');
    plt.show();

def plot_station_different_pollutants(pollutant_station_datas, station_id, n_days=7, begin_time=72):
    end_time = begin_time + 24 * n_days

    for pollutant_key in pollutant_station_datas:
        pollutant_station_data = pollutant_station_datas[pollutant_key][station_id]
        pollutant_station_data = pollutant_station_data.loc[pollutant_station_data['daytime'] >= begin_time and pollutant_station_data['daytime'] <= end_time]
        plt.plot(pollutant_station_data['daytime'].values, pollutant_station_data[pollutant_key].values)
        plt.legend(pollutant_key)
    plt.title('Pollutants values at station ' + station_id)
    plt.axis(x='time (h)', y='pollutant concentration')
    plt.show()

def plot_zone_station_values(zone_station_pollutant_datas, zone_id, pollutant, n_days=7, begin_time=72):
    end_time = begin_time + 24 * n_days

    station_pollutant_datas = zone_station_pollutant_datas[zone_id]

    for station_id_key in station_pollutant_datas:
        station_datas = station_pollutant_datas[station_id_key]
        if pollutant in station_datas.keys():
            data = station_datas[pollutant]
            data = data.loc[data['daytime'] <= end_time]
            plot_label = 'station ' + str(station_id_key)
            plt.plot(data['daytime'].values, data['TARGET'].values, label=plot_label)
        else:
            print 'station ', station_id_key, ' has no data for pollutant ', pollutant
    plt.legend()
    plt.title(pollutant + ' values at stations in zone ' + str(zone_id))
    plt.axis(x='time (h)', y='pollutant concentration')
    plt.show()

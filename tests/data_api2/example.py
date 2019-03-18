import data_api2 as api
import datetime

def main():

    pulse_id = api.get_pulse_id_from_timestamp(datetime.datetime.now()-datetime.timedelta(seconds=20))
    print(pulse_id)

    query = api.construct_data_query(channels=['SF-IMAGEBUFFER/SLAAR21-LCAM-C511:FPICTURE'],
                                     start=7945618542,
                                     end=7945618542)

    # query = api.construct_data_query(channels=['sf-imagebuffer/SLAAR21-LCAM-C511:FPICTURE'],
    #                                  start=7928427268,
    #                                  end=7928427268)

    data = api.get_data(query)
    #
    # for c in data:
    #     if c["channel"]["name"] == "xxx":
    #         #do something
    #         values = c["data"]
    #         pass
    #
    # values = data["xxx"]

    print(data)

    data = api.as_dict(data)
    # print(data["sf-imagebuffer/SLAAR21-LCAM-C511:FPICTURE"])

    print(data['SF-IMAGEBUFFER/SLAAR21-LCAM-C511:FPICTURE'])
    print("-"*10)
    print(data['SLAAR21-LCAM-C511:FPICTURE'])

    for i in data:
        print(i)


if __name__ == '__main__':
    main()

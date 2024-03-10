from datetime import datetime

def add_schedule(entry):
    with open('schedule.txt', 'a') as f:
        f.write(f'{entry}\n')

def get_schedule():
    today = datetime.now().strftime('%Y-%m-%d %H:%M')
    future_events = []
    todays_events = []
    with open('schedule.txt', 'r') as f:
        lines = f.readlines()
        cur_date, cur_time = today[:10], today[11:18]
        for line in lines:
            date, time, event = line[:10], line[11:16], line[17:-1]
            if int(date[8:10]) - int(cur_date[8:10]) == 0:
                todays_events.append(f'{time}\n{event}')
                future_events.append(line)
            elif int(date[8:10]) - int(cur_date[8:10]) > 0:
                future_events.append(line)
    with open('schedule.txt', 'w') as f:
        for line in future_events:
            f.write(line)
        f.close()
    return todays_events

from skyfield.api import Star
import pytz
from datetime import timedelta

def get_sky_forecast(anchor_row, observer, ts):
    t_now = ts.now()
    target_star = Star(ra_hours=anchor_row['ra'], dec_degrees=anchor_row['dec'])
    astrometric = observer.at(t_now).observe(target_star)
    alt, az, _ = astrometric.apparent().altaz()
    compass_dir = ["N", "NE", "E", "SE", "S", "SW", "W", "NW", "N"][int(round((az.degrees % 360) / 45))]
    
    is_visible = alt.degrees > 0
    if is_visible:
        print(f"    -> STATUS: VISIBLE NOW! Look {compass_dir} and tilt your head up {alt.degrees:.0f} degrees.")
    else:
        print(f"    -> STATUS: BELOW HORIZON. Simulating orbital rotation...")
        temp_t = t_now
        for _ in range(24 * 4): 
            temp_t = ts.utc(temp_t.utc_datetime() + timedelta(minutes=15))
            a, _, _ = observer.at(temp_t).observe(target_star).apparent().altaz()
            if a.degrees > 0:
                local_tz = pytz.timezone('US/Eastern')
                local_time = temp_t.utc_datetime().replace(tzinfo=pytz.utc).astimezone(local_tz)
                print(f"    -> STATUS: It will crest the horizon next at {local_time.strftime('%I:%M %p on %b %d')}.")
                break
                
    return alt.degrees, az.degrees, compass_dir
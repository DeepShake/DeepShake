## time-series-balancing: 
    Train: X normalized with 15s history, Y(10s future) normalized with 15s X history
    Val: X normalized with 15s history, Y(10s future) normalized with 15s X history
    
## time-series-balancing-v2: 
    Train: X normalized with 15s history, Y(10s future) normalized with 60s quake history
    Val: X normalized with 15s history, Y(10s future) normalized with 15s X history

## time-series-balancing-v3: 
    Train: X normalized (TIME+STATIONS) with 15s history, Y(10s future) normalized (TIME+STATIONS) with 15s quake history
    Val: X normalized (TIME+STATIONS) with 15s history, Y(10s future) normalized (TIME+STATIONS) with 15s X history
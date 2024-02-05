### Changing to datetime axis

Bokeh can only display charts at milliseconds granularity only
so, we need to convert all our timestamps

```python
df["timestamp"] = (
    pd.to_datetime(
            combined_datetime_str, format="%Y-%m-%d %H:%M:%S.%f"
        ).astype(np.int64)
        // 1000000
```

Since `np.int64` converts everything to nanoseconds, we convert them back to milliseconds wherever we use the timestamp

Once this is done, we can just apply the default formatter

```python
overlay.opts(
    title="Full Day 10-Deep Scatter Plot with Dynamic Tooltip Depending on Zoom Level",
    width=1200,
    height=600,
    xlabel="Timestamp",
    ylabel="Price",
    bgcolor="#002333",
    yformatter=NumeralTickFormatter(format="0,0"),
    xformatter=DatetimeTickFormatter(),  # x-axis as timestamp
    )
```

I ran this with `panel serve ts_vis.py` 

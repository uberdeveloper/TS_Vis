import numpy as np
import pandas as pd
import panel as pn
import panel.widgets as pnw
import param
import holoviews as hv
from holoviews.operation.datashader import datashade, dynspread
from holoviews import opts
from bokeh.models.formatters import (
    BasicTickFormatter,
    NumeralTickFormatter,
    DatetimeTickFormatter,
)
import colorcet
import datetime as dt
from io import StringIO
import os

hv.extension("bokeh")


def filter_points(points, x_range, y_range):
    if x_range is None or y_range is None:
        return points
    return points[x_range, y_range]


def hover_points(points, threshold=200):
    if len(points) > threshold:
        return points.iloc[:0]
    return points


def label_points(points, i, j, threshold=200):
    if len(points) > threshold:
        return hv.Labels(points.iloc[:0], vdims=[i, j])
    return hv.Labels(points, vdims=[i, j]).opts(
        text_color="white", text_font_size="10pt"
    )


def clean_Tag50(s):
    """
    allows one or multiple tag50s
    assume tag50s entered are separated by comma or space
    always return tag50 in uppercase
    """
    return [i for i in s.upper().replace(" ", ",").split(",") if i]


def clean_OrderID(s):
    """
    allows one or multiple order ids
    assume order ids entered are separated by comma or space
    """
    return [i for i in s.replace(" ", ",").split(",") if i]


# change the width of table columns
def apply_format(plot, element):
    plot.handles["table"].autosize_mode = "fit_columns"


# customize bids/asks (blues/yellows) colors
color_list = colorcet.bky[0::8]
# select the first 10 (10 levels) blues from the beginning of the colorcet.bky palette
custom_blues = colorcet.bky[0::8][:10]
# select the last 10 (10 levels) yellows from the end of the colorcet.bky palette
custom_yellows = colorcet.bky[0::8][::-1][:10]
# swatch(name='custom_blues', cmap=custom_blues)
color_sequence = custom_blues + custom_yellows

# cutomize color palette for order highlights,
# there are 17 message types from OE
highlight_palette = [
    "#00ffff",
    "#00ff00",
    "#ff0000",
    "#ffff00",
    "#0000ff",
    "#ff6200",
    "#9ee2ff",
    "#00c846",
    "#ff28fd",
    "#b8ba01",
    "#f4bfb1",
    "#009e7c",
    "#ffa52f",
    "#afa5ff",
    "#a0e491",
    "#c6a5c1",
    "#ff9070",
]

# set default styles and plot options
opts.defaults(
    opts.Points(
        tools=["tap", "wheel_zoom", "hover"],
        active_tools=["wheel_zoom", "box_zoom"],
        color=hv.Cycle(values=color_sequence),
        marker="square",
        alpha=0.7,
        hover_alpha=0.2,
        size=20,
        clipping_colors={"NaN": (0, 0, 0, 0)},
    )
)

pn.param.ParamMethod.loading_indicator = True


class order_book_viewer(param.Parameterized):
    prices = [
        "bid_price_1",
        "bid_price_2",
        "bid_price_3",
        "bid_price_4",
        "bid_price_5",
        "bid_price_6",
        "bid_price_7",
        "bid_price_8",
        "bid_price_9",
        "bid_price_10",
        "ask_price_1",
        "ask_price_2",
        "ask_price_3",
        "ask_price_4",
        "ask_price_5",
        "ask_price_6",
        "ask_price_7",
        "ask_price_8",
        "ask_price_9",
        "ask_price_10",
    ]
    qtys = [
        "bid_qty_1",
        "bid_qty_2",
        "bid_qty_3",
        "bid_qty_4",
        "bid_qty_5",
        "bid_qty_6",
        "bid_qty_7",
        "bid_qty_8",
        "bid_qty_9",
        "bid_qty_10",
        "ask_qty_1",
        "ask_qty_2",
        "ask_qty_3",
        "ask_qty_4",
        "ask_qty_5",
        "ask_qty_6",
        "ask_qty_7",
        "ask_qty_8",
        "ask_qty_9",
        "ask_qty_10",
    ]
    counts = [
        "bid_cnt_1",
        "bid_cnt_2",
        "bid_cnt_3",
        "bid_cnt_4",
        "bid_cnt_5",
        "bid_cnt_6",
        "bid_cnt_7",
        "bid_cnt_8",
        "bid_cnt_9",
        "bid_cnt_10",
        "ask_cnt_1",
        "ask_cnt_2",
        "ask_cnt_3",
        "ask_cnt_4",
        "ask_cnt_5",
        "ask_cnt_6",
        "ask_cnt_7",
        "ask_cnt_8",
        "ask_cnt_9",
        "ask_cnt_10",
    ]

    ## widget params ##
    Instrument = param.String("")
    Date = param.CalendarDate(
        default=None, bounds=(dt.date(2000, 1, 1), dt.date(2099, 12, 31))
    )
    submit = param.Action(lambda x: x.param.trigger("submit"), label="Submit")
    OrderID = param.String(default="", doc="Type Order ID", label="Order ID")
    Tag50 = param.String(default="", doc="Type Tag50", label="Tag50")
    search = param.Action(lambda x: x.param.trigger("search"), label="Search")
    highlight = param.Boolean(default=False, label="Highlight Orders")
    armada = []
    oe_data = []
    oe_search_result = []
    oe_merge_result = []

    # base directory where the data are saved
    basedir = 'C:/Users/John/cftc/'
    print(basedir)

    ## ARMADA DATA FROM CACHE OR QUERY FROM REDSHIFT AND CACHE ##
    def get_data(self):
        global df
        # initial value
        if len(self.Instrument) == 0:
            self.armada = []

        # read data from cache if exists; query and save data otherwise
        else:
            # construct cache file name dynamically
            #             armada_cache_file = self.basedir + f'obv_armada_{self.Instrument.upper()}_{self.Date.isoformat()}.parquet'
            armada_cache_file = (
                self.basedir
                + f"obv_mbp_{self.Instrument.upper()}_{self.Date.isoformat()}.parquet"
            )

            # check if the cache file exists already, and read data from cache file if it exists
            df = pd.read_parquet(armada_cache_file, engine="fastparquet")

            combined_datetime_str = df["date"] + " " + df["time"]
            df["timestamp0"] = (
                df["time"]
                .str.replace(":", "")
                .str.replace(".", "")
                .astype(str)
                .astype(np.int64)
            )
            df["timestamp1"] = pd.to_datetime(
                combined_datetime_str, format="%Y-%m-%d %H:%M:%S.%f"
            )
            df["timestamp"] = (
                pd.to_datetime(
                    combined_datetime_str, format="%Y-%m-%d %H:%M:%S.%f"
                ).astype(np.int64)
                // 1000000
            )
            self.armada = df

    ## OE DATA FROM CACHE OR QUERY FROM REDSHIFT AND CACHE ##
    def get_oe_data(self):
        # initial value
        if len(self.Instrument) == 0:
            self.oe_data = []

        # read data from cache if exists, query and save data otherwise
        else:
            # construct cache file name dynamically
            oe_cache_file = (
                self.basedir
                + f"obv_oe_{self.Instrument.upper()}_{self.Date.isoformat()}.parquet"
            )

            # check if the cache file exists already, and read data from cache file if it exists
            if os.path.exists(oe_cache_file):
                self.oe_data = pd.read_parquet(oe_cache_file, engine="fastparquet")

            # if cache file does not exist, send query and process query results
            else:
                try:
                    oe_key_cols = [
                        "exchange",
                        "business_date",
                        "calendar_date",
                        '"time"',
                        "sequence_number",
                        '"group"',
                        "instrument",
                        "reg_message_type",
                        "order_id",
                        "tag50",
                        "buy_sell",
                        "price",
                        "quantity",
                    ]

                    sql = cp.GenSQL()
                    query = sql.oe_query(
                        instruments=self.Instrument.upper(),
                        start_date_or_list=[self.Date.isoformat()],
                        usecols=oe_key_cols,
                    )
                    puller = cp.CloudPuller()
                    oe = puller.compute_pull_and_load(query, clear_s3_data=True)

                    # sort pulled OE data
                    oe = oe.sort_values(
                        by=["calendar_date", "time", "sequence_number"]
                    ).reset_index(drop=True)

                    # change any date columns type to str
                    oe[["calendar_date", "business_date"]] = oe[
                        ["calendar_date", "business_date"]
                    ].astype(str)

                    # due to NAs in the quantity column (pd.NA in int32 column type) from OE, oe_search_result does not display in hv.Table correctly
                    # fill pd.NA with 0
                    oe["quantity"] = oe["quantity"].fillna(0)

                    # update self.oe_data
                    self.oe_data = oe

                    # save query results to the cache directory if there are results, do not save if user enters an invalid Instrument/Date or no data returned for the Instrument/Date combination
                    if len(oe) > 0:
                        # check if the cache file path exists or not, create a file path if not, and save processed query results to the cache directory
                        if not os.path.isdir(self.basedir):
                            os.makedirs(self.basedir)
                        oe.to_parquet(oe_cache_file, engine="fastparquet")

                except:
                    self.oe_data = []

    @param.depends("submit", watch=True)
    def update_data(self):
        # clear previous data when making changes in Instrument/Date, and set params to the initial state
        self.armada = []
        self.oe_data = []
        self.oe_search_result = []
        self.oe_merge_result = []
        self.OrderID = ""
        self.Tag50 = ""
        self.highlight = False

        # to update self.armada
        self.get_data()

        # to update self.oe_data only if armada exists,
        # OE is available from '2018-10-01', leave it blank otherwise
        if len(self.armada) > 0 and self.Date.isoformat() >= "2018-10-01":
            self.get_oe_data()
        else:
            pass

    @param.depends("submit")
    def message_pane(self):
        if len(self.Instrument) == 0:
            return pn.pane.Alert(
                "Enter an instrument and date to start.", alert_type="secondary"
            )
        elif len(self.armada) == 0:
            return pn.pane.Alert(
                "There are no matches for your search. Please try again.",
                alert_type="danger",
            )
        else:
            return pn.pane.Alert("", visible=False)

    ## SEARCH OE DATA ##
    def oe_search(self):
        oe = self.oe_data

        if len(oe) > 0:
            if self.Tag50 != "" and self.OrderID != "":
                oe2 = oe[
                    (oe["tag50"].isin(clean_Tag50(self.Tag50)))
                    & (oe["order_id"].isin(clean_OrderID(self.OrderID)))
                ]
            elif self.Tag50 == "" and self.OrderID != "":
                oe2 = oe[oe["order_id"].isin(clean_OrderID(self.OrderID))]
            elif self.Tag50 != "" and self.OrderID == "":
                oe2 = oe[oe["tag50"].isin(clean_Tag50(self.Tag50))]
            else:
                oe2 = []
        else:
            oe2 = []

        return oe2

    ## SEARCH OE TABLE ##
    @param.depends("search", "submit")
    def oe_search_table(self):
        self.oe_search_result = self.oe_search()

        # display up to 5000 records in the oe search table
        return hv.Table(self.oe_search_result[:5000]).opts(
            width=1600, hooks=[apply_format]
        )

    ## MERGED DATA ##
    def merged_data(self):
        ar = self.armada
        oe2 = self.oe_search_result

        if len(oe2) > 0:
            # key columns from ARMADA to be merged with OE
            unique_cols = ["time", "timestamp"] + ar.columns.tolist()[3:-1]

            me = pd.merge(oe2, ar[unique_cols], on="time", how="left")

            # reset index, starts from 1
            me.reset_index(drop=True, inplace=True)
            me.index += 1

        else:
            me = []

        return me

    ## DOWNLOAD MERGED DATA ##
    def merged(self):
        me = self.oe_merge_result
        if len(me) > 0:
            sio = StringIO()
            me.to_csv(sio)
            sio.seek(0)
        else:
            sio = dict()
        return sio

    ## MERGED TABLE ##
    @param.depends("search", "submit")
    def merged_table(self):
        self.oe_merge_result = self.merged_data()

        # display up to 5000 records in the merged table
        return hv.Table(self.oe_merge_result[:5000]).opts(
            width=1600, hooks=[apply_format]
        )

    ## MAIN PLOT ##
    @param.depends("submit", "highlight")
    def mplot(self):
        if len(self.armada) == 0:
            return ""
        else:
            price_series = []
            price_hover = []
            price_label = []
            df = self.armada
            oe = self.oe_data
            for i, value in enumerate(self.prices):
                kdims = ["timestamp", value]
                vdims = ["instrument", "time", self.qtys[i], self.counts[i]]
                points = hv.Points(df, kdims, vdims)
                range_stream = hv.streams.RangeXY(source=points)
                filtered = points.apply(filter_points, streams=[range_stream])
                shaded = datashade(filtered, streams=[range_stream])
                hover = filtered.apply(hover_points)
                labeled = filtered.apply(label_points, i=self.qtys[i], j=self.counts[i])

                price_series.append(shaded)
                price_hover.append(hover)
                price_label.append(labeled)

            # Expression to create plot based on list of objects
            expression = [
                f"price_series[{i}]"
                + " * "
                + f"price_hover[{i}]"
                + " * "
                + f"price_label[{i}]"
                for i in range(0, len(self.prices))
            ]
            joined_expression = " * ".join([i for i in expression])
            dynamic_hover = eval(joined_expression)

            # trade highlight function
            def oe_trades_in_streamed_range(x_range, y_range, threshold=2000):
                # find and highlight trades from OE on the armada plot in zoom-in view
                if x_range and y_range:
                    # highlight trades when the number of timestamp updates <= threshold when zoomed in, hide trades otherwise
                    df1 = points[x_range, y_range]
                    if len(df1) > 0 and len(df1) < threshold:
                        # get time range from armada timestamp range when zoomed in
                        df2 = points[x_range, y_range].data[["timestamp", "time"]]
                        time_min = df2.time.min()
                        time_max = df2.time.max()
                        # a trade: column reg_message_type == "FILL" in OE
                        # map the time range from armada to OE time
                        oe2 = oe[
                            (oe.time >= time_min)
                            & (oe.time <= time_max)
                            & (oe.reg_message_type == "FILL")
                        ][["time", "price", "order_id", "buy_sell", "quantity"]]
                        # merge trades happened in the time range from OE with timestamp_index from armada, now we have timestamp_indexes & trade prices to plot
                        df_trade = df2.merge(oe2, on="time", how="inner")
                        trade_points = hv.Points(df_trade, ["timestamp", "price"]).opts(
                            tools=[],
                            marker="square",
                            color="#FFFF00",
                            line_width=3,
                            line_alpha=1,
                            alpha=1,
                            hover_alpha=0.2,
                            size=20,
                            width=600,
                            height=400,
                        )
                        trade_labels = hv.Labels(
                            df_trade, kdims=["timestamp", "price"], vdims=["quantity"]
                        ).opts(text_color="black", text_font_size="10pt")
                    else:
                        trade_points = hv.Points([])
                        trade_labels = hv.Labels([])
                else:
                    trade_points = hv.Points([])
                    trade_labels = hv.Labels([])

                return trade_points * trade_labels

            rangexy = hv.streams.RangeXY(source=points)
            trades = hv.DynamicMap(oe_trades_in_streamed_range, streams=[rangexy])

            # default highlight value is False = highlight box is not checked
            if not self.highlight:
                overlay = dynamic_hover * trades
                overlay.opts(
                    title="Full Day 10-Deep Scatter Plot with Dynamic Tooltip Depending on Zoom Level",
                    width=1200,
                    height=600,
                    xlabel="Timestamp",
                    ylabel="Price",
                    bgcolor="#002333",
                    # apply custom format to yformatter to display time instead of time index
                    # use this link for reference https://docs.bokeh.org/en/3.1.0/docs/reference/models/formatters.html
                    yformatter=NumeralTickFormatter(format="0,0"),
                    xformatter=DatetimeTickFormatter(),  # x-axis as timestamp
                )
            else:
                df_orders_kdims = ["timestamp", "price"]
                df_orders_vdims = [
                    "business_date",
                    "time",
                    "sequence_number",
                    "reg_message_type",
                    "order_id",
                    "tag50",
                    "buy_sell",
                    "quantity",
                ]
                # highlight up to 100 records in the chart
                highlighted = hv.Points(
                    self.oe_merge_result[:100], df_orders_kdims, df_orders_vdims
                )
                overlay = (
                    dynamic_hover
                    * trades
                    * highlighted.opts(
                        line_color="reg_message_type",
                        line_width=3,
                        line_alpha=1,
                        cmap=highlight_palette,
                        size=20,
                        fill_alpha=0,
                    )
                )
                overlay.opts(
                    title="Full Day 10-Deep Scatter Plot with Dynamic Tooltip Depending on Zoom Level",
                    width=1200,
                    height=600,
                    xlabel="Timestamp Index",
                    ylabel="Price",
                    bgcolor="#002333",
                    # apply custom format to yformatter to display time instead of time index
                    # use this link for reference https://docs.bokeh.org/en/3.1.0/docs/reference/models/formatters.html
                    yformatter=NumeralTickFormatter(format="0,0"),
                    xformatter=DatetimeTickFormatter(),  # x-axis as timestamp
                    ),  # x-axis as timestamp index separated by comma
                )

            def clicked_ar(x, y):
                return hv.Table(
                    df.loc[
                        df.timestamp == round(x, 0),
                        ["timestamp", "date", "time"] + df.columns.tolist()[3:-1],
                    ]
                ).relabel("Order Book Table")

            def clicked_oe(x, y):
                if len(oe) > 0:
                    df2 = df[df.timestamp == round(x, 0)]
                    click = df2["time"].to_string()
                    clicked_time = click[-18:]
                    oe2 = oe.loc[oe["time"] == clicked_time]
                else:
                    oe2 = []
                return hv.Table(oe2).relabel("OE Table")

            tap_point = hv.streams.Tap(source=points, x=np.nan, y=np.nan)
            ar_table = hv.DynamicMap(clicked_ar, streams=[tap_point]).opts(
                width=1220, height=80, hooks=[apply_format]
            )
            oe_table = hv.DynamicMap(clicked_oe, streams=[tap_point]).opts(
                width=1220, height=125, hooks=[apply_format]
            )

            # streaming start/end time in hv.Div()
            def start_time(x_range, y_range):
                if x_range and y_range:
                    sel = points[x_range, y_range].data
                    if len(sel) > 0:
                        start_time = sel["date"].iloc[0] + " " + sel["time"].iloc[0]
                        div = hv.Div("<div'>" + "Time Start " + start_time + "<div>")
                    else:
                        div = hv.Div("<div'>" + " " + "<div>")
                else:
                    div = hv.Div("<div'>" + " " + "<div>")

                return div.opts(height=20, width=300)

            def end_time(x_range, y_range):
                if x_range and y_range:
                    sel = points[x_range, y_range].data
                    if len(sel) > 0:
                        end_time = sel["date"].iloc[-1] + " " + sel["time"].iloc[-1]
                        div = hv.Div("<div'>" + "Time End " + end_time + "<div>")
                    else:
                        div = hv.Div("<div'>" + " " + "<div>")
                else:
                    div = hv.Div("<div'>" + " " + "<div>")

                return div.opts(height=20, width=300)

            rangexy = hv.streams.RangeXY(source=points)
            start = hv.DynamicMap(start_time, streams=[rangexy])
            end = hv.DynamicMap(end_time, streams=[rangexy])

            layout = pn.Column(
                overlay,
                pn.Row(start, pn.Spacer(width=650), end),
                ar_table,
                oe_table,
            )

            return layout

    def panel(self):
        return pn.Column(
            "# Order Book Data Viewer v1.1 DEV - MBP UAT Test",
            pn.Row(
                self.param.Instrument,
                pn.Column(
                    self.param.Date,
                    "ARMADA is available from 2013-01-01 to 2018-12-31",
                    "MBP UAT is available from 2022-05-22 to 2022-05-30",
                ),
                pn.Column(self.param.submit, margin=(18, 0, 0, 0)),
            ),
            self.message_pane,
            self.mplot,
            "### **Search Order ID and/or Tag 50**",
            pn.Row(
                self.param.OrderID,
                self.param.Tag50,
                pn.Column(self.param.search, margin=(18, 0, 0, 0)),
                pn.Column(self.param.highlight, margin=(25, -150, 0, 0)),
                pn.widgets.FileDownload(
                    callback=self.merged,
                    filename="Merged_OE_OrderBook.csv",
                    margin=(23, 0, 0, 10),
                ),
            ),
            self.oe_search_table,
            "### **Merged OE with Order Book Data**",
            self.merged_table,
        )


# create instance of dashboard
obv = order_book_viewer()
obv.panel().servable()
# ESH9 2018-10-05

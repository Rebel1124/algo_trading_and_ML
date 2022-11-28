## UTOR: FinTech Bootcamp - Project 2: Trading Strategy Optimizer

if __name__ == '__main__':

    # Import libraries
    import numpy as np
    import pandas as pd
    from pandas.tseries.offsets import DateOffset
    pd.set_option('mode.chained_assignment', None)
    pd.core.common.is_list_like = pd.api.types.is_list_like
    import time
    import datetime
    from datetime import date, timedelta
    import plotly.express as px
    import plotly.graph_objects as go
    import plotly.io as pio
    import streamlit as st
    from PIL import Image

    import pandas_ta as ta    # https://github.com/twopirllc/pandas-ta



    # Standard/Set Inputs

    # Time Interval
    timePeriod="max"
    daily="1d"

    # Portfolio Metrics
    initial = 10000
    riskFree = 0.01


    # Streamlit Setup
    # STreamlit Asset Inputs
        
    image = Image.open('arrow.png')    
    
    cola, colb = st.columns([1, 4.5])
    cola.image(image, use_column_width=True)
    
    colb.markdown("<h1 style='text-align: left; color: Purple; padding-left: 0px; font-size: 60px'>ARROW-UP CAPITAL</h1>", unsafe_allow_html=True)
    
    st.markdown("<h2 style='text-align: left; color: teal; padding-left: 45px; font-size: 50px'>Trading Strategy Optimizer</h2>", unsafe_allow_html=True)
    
    st.markdown(" ")

    st.markdown("<h3 style='text-align: left; color: #872657; padding-left: 0px; font-size: 30px'><b>Asset Inputs<b></h3>", unsafe_allow_html=True)
    
    # User to choose asset
    colc, cold, cole, colf = st.columns([1, 1, 1, 1])

    assetCodesList = ['CL=F', 'GC=F', '^RUT', '^GSPC', 'EURUSD=X', 'GBPJPY=X', 'BTC-USD', 'ETH-USD']
    assetNamesList = ['crudeOil', 'Gold', 'Russel2000', 'S&P500', 'EUR-USD', 'GBP-JPY', 'BTC-USD', 'ETH-USD']
    
    assetNames = colc.selectbox('Asset Selection', assetNamesList, index=0)
    index = assetNamesList.index(assetNames)
    assetCodes = assetCodesList[index]

    # Date settings
    start_date = cold.date_input('Start Date',date(2006,6,30)).strftime("%Y-%m-%d")
    end_date = cole.date_input('End Date',date(2022,11,1)).strftime("%Y-%m-%d")
    training_months = colf.number_input('Training Months', min_value=42, max_value=1260, step=1, value=145)

    # streamlit sidebar and strategy indicator settings setup
    
    chessImage = Image.open('chess.png')
    st.sidebar.image(chessImage, use_column_width=True)
    st.sidebar.markdown("<h3 style='text-align: left; color: #872657; padding-left: 0px; font-size: 30px'><b>Trading Strategy Inputs<b></h3>", unsafe_allow_html=True)
    st.sidebar.markdown(' ')

    # Weekly Indicators
    st.sidebar.markdown("<h4 style='text-align: left; color: #551A8B; padding-left: 0px; font-size: 20px'><b>Weekly Indicator Settings<b></h4>", unsafe_allow_html=True)
    weeklyEMAShort = st.sidebar.number_input('Weekly Short EMA', min_value=3, max_value=12, step=1, value=5)
    weeklyEMALong = st.sidebar.number_input('Weekly Long EMA', min_value=9, max_value=18, step=1, value=13)
    weeklyADX = st.sidebar.number_input('Weekly ADX', min_value=3, max_value=9, step=1, value=6)

    st.sidebar.markdown(' ')
    st.sidebar.markdown(' ')

    # Daily Indicators
    st.sidebar.markdown("<h4 style='text-align: left; color: #551A8B; padding-left: 0px; font-size: 20px'><b>Daily Indicator Settings<b></h4>", unsafe_allow_html=True)
    dailyEMAShort = st.sidebar.number_input('Daily Short EMA', min_value=9, max_value=18, step=1, value=12)
    dailyEMALong = st.sidebar.number_input('Daily Long EMA', min_value=18, max_value=32, step=1, value=26)
    elderRayLength = st.sidebar.number_input('Elder-Ray', min_value=6, max_value=18, step=1, value=12)
    bbandsLength = st.sidebar.number_input('Bollinger Bands', min_value=12, max_value=32, step=1, value=21)
    macdFast = st.sidebar.number_input('MACD Fast', min_value=6, max_value=18, step=1, value=12)
    macdSlow = st.sidebar.number_input('MACD Slow', min_value=18, max_value=32, step=1, value=26)     
    macdSignal = st.sidebar.number_input('MACD Signal', min_value=6, max_value=21, step=1, value=9)


    # Indicator Names

    # Number of Assets
    count = len(assetCodes)


    # Weekly Indicators
    weeklyEMAShortIndicatorName = 'EMA_'+str(weeklyEMAShort)
    newWeeklyEMAShortIndicatorName = 'weeklyEMAShort'

    weeklyEMALongIndicatorName = 'EMA_'+str(weeklyEMALong)
    newWeeklyEMALongIndicatorName = 'weeklyEMALong'

    weeklyADXName = 'ADX_'+str(weeklyADX)
    newWeeklyADXName = 'weeklyADX'


    # Daily Indicators
    dailyEMAShortIndicatorName = 'EMA_'+str(dailyEMAShort)
    newDailylyEMAShortIndicatorName = 'dailyEMAShort'

    dailyEMALongIndicatorName = 'EMA_'+str(dailyEMALong)
    newDailyEMALongIndicatorName = 'dailyEMALong'

    bullPowerIndicatorName = 'BULLP_'+str(elderRayLength)
    newBullPowerIndicatorName = 'BullPower'

    bearPowerIndicatorName = 'BEARP_'+str(elderRayLength)
    newBearPowerIndicatorName = 'BearPower'

    bollingerLowerIndicatorName = 'BBL_'+str(bbandsLength)+'_2.0'
    newBollingerLowerIndicatorName = 'lowerBB'

    bollingerMiddleIndicatorName = 'BBM_'+str(bbandsLength)+'_2.0'
    newBollingerMiddleIndicatorName = 'middleBB'

    bollingerUpperIndicatorName = 'BBU_'+str(bbandsLength)+'_2.0'
    newBollingerUpperIndicatorName = 'upperBB'

    bollingerStdIndicatorName = 'BBB_'+str(bbandsLength)+'_2.0'
    newBollingerStdIndicatorName = '2stdBB'

    macdIndicatorName = "MACD_"+str(macdFast)+"_"+str(macdSlow)+"_"+str(macdSignal)
    newMACDIndicatorName = "MACDline"

    macdHistogramIndicatorName = "MACDh_"+str(macdFast)+"_"+str(macdSlow)+"_"+str(macdSignal)
    newMACDHistogramIndicatorName = "MACDHistogram"

    macdSignalIndicatorName = "MACDs_"+str(macdFast)+"_"+str(macdSlow)+"_"+str(macdSignal)
    newMACDSignalIndicatorName = "MACDSignal"


    ####################################################################################################################################
    ################################### Define Functions ###############################################################################

    # Function to calculate weekly Indicators

    @st.cache(allow_output_mutation=True)
    def weeklyIndicators(assetCodes, timePeriod, daily, start_date, end_date, weeklyEMAShort, weeklyEMALong, weeklyADX,
                        weeklyEMAShortIndicatorName, weeklyEMALongIndicatorName, weeklyADXName,
                        newWeeklyEMAShortIndicatorName,newWeeklyEMALongIndicatorName, newWeeklyADXName):
        
        # Get Weekly Asset Data
        dfWeekly = pd.DataFrame()
        dfWeekly = dfWeekly.ta.ticker(assetCodes, period=timePeriod, interval=daily)
        dfWeekly = dfWeekly[(dfWeekly.index > start_date) & (dfWeekly.index < end_date)]
        dfWeekly = dfWeekly[(dfWeekly.Close > 0)]
        dfWeekly = dfWeekly[(dfWeekly.index.dayofweek == 0)]

        # Create Custom Weekly Strategy
        CustomStrategyWeekly = ta.Strategy(
            name="Weekly Indicators",
            description="Weekly EMA and ADX Indicators",
            ta=[
                {"kind": "ema", "length": weeklyEMAShort},
                {"kind": "ema", "length": weeklyEMALong},
                {"kind": "adx", "length": weeklyADX},
            ]
        )
        
        # Run "Custom Weekly Strategy"
        dfWeekly.ta.strategy(CustomStrategyWeekly)
        dfWeekly=dfWeekly.dropna()
        algoDataWeekly = dfWeekly[[weeklyEMAShortIndicatorName, weeklyEMALongIndicatorName, weeklyADXName]]

        algoDataWeekly = algoDataWeekly.rename({weeklyEMAShortIndicatorName: newWeeklyEMAShortIndicatorName,
                                                weeklyEMALongIndicatorName: newWeeklyEMALongIndicatorName,
                                                weeklyADXName: newWeeklyADXName}, axis=1)
        
        
        return algoDataWeekly


    # Function to calculate daily Indicators

    @st.cache(allow_output_mutation=True)
    def dailyIndicators(assetCodes, timePeriod, daily, start_date, end_date, dailyEMAShort, dailyEMALong, elderRayLength,
                        bbandsLength, macdFast, macdSlow, macdSignal, dailyEMAShortIndicatorName, dailyEMALongIndicatorName,
                        bullPowerIndicatorName, bearPowerIndicatorName, bollingerLowerIndicatorName, 
                        bollingerMiddleIndicatorName, bollingerUpperIndicatorName, bollingerStdIndicatorName, macdIndicatorName,
                        macdHistogramIndicatorName, macdSignalIndicatorName, newDailylyEMAShortIndicatorName,
                        newDailyEMALongIndicatorName, newBullPowerIndicatorName, newBearPowerIndicatorName, 
                        newBollingerLowerIndicatorName, newBollingerMiddleIndicatorName, newBollingerUpperIndicatorName,
                        newBollingerStdIndicatorName, newMACDIndicatorName, newMACDHistogramIndicatorName,
                        newMACDSignalIndicatorName):
        
        # Get Daily Asset Data

        dfDaily = pd.DataFrame()
        dfDaily = dfDaily.ta.ticker(assetCodes, period=timePeriod, interval=daily)
        dfDaily = dfDaily[(dfDaily.index > start_date) & (dfDaily.index < end_date)]
        dfDaily = dfDaily[(dfDaily.Close > 0)]

        # Use the pct_change function to generate returns from close prices
        dfDaily["ActualReturns"] = dfDaily["Close"].pct_change()
        
        # Drop all NaN values from the DataFrame
        dfDaily = dfDaily.dropna()

        # Initialize the new Signal column
        dfDaily['Signal'] = 0.0

        # When Actual Returns are greater than or equal to 0, generate signal to buy asset long
        dfDaily.loc[(dfDaily['ActualReturns'] >= 0), 'Signal'] = 1

        # When Actual Returns are less than 0, generate signal to sell asset short
        dfDaily.loc[(dfDaily['ActualReturns'] < 0), 'Signal'] = -1


        # Create your own Custom Strategy
        CustomStrategyDaily = ta.Strategy(
            name="Daily Indicators",
            description="daily Trading Indicators",
            ta=[
                {"kind": "ema", "length": dailyEMAShort},
                {"kind": "ema", "length": dailyEMALong},
                {"kind": "eri", "length": elderRayLength},
                {"kind": "bbands", "length": bbandsLength},
                {"kind": "macd", "fast": macdFast, "slow": macdSlow, "signal": macdSignal},
            ]
        )


        # Run "Custom Daily Strategy"
        dfDaily.ta.strategy(CustomStrategyDaily)
        dfDaily=dfDaily.dropna()
        algoDataDaily = dfDaily[['Close', 'ActualReturns','Signal',dailyEMAShortIndicatorName, dailyEMALongIndicatorName,
                            bullPowerIndicatorName, bearPowerIndicatorName, bollingerLowerIndicatorName,
                            bollingerMiddleIndicatorName, bollingerUpperIndicatorName, bollingerStdIndicatorName,
                            macdIndicatorName, macdHistogramIndicatorName, macdSignalIndicatorName]]

        algoDataDaily = algoDataDaily.rename({dailyEMAShortIndicatorName: newDailylyEMAShortIndicatorName,
                                dailyEMALongIndicatorName: newDailyEMALongIndicatorName,
                                bullPowerIndicatorName: newBullPowerIndicatorName,
                                bearPowerIndicatorName: newBearPowerIndicatorName,
                                bollingerLowerIndicatorName: newBollingerLowerIndicatorName,
                                bollingerMiddleIndicatorName: newBollingerMiddleIndicatorName,
                                bollingerUpperIndicatorName: newBollingerUpperIndicatorName,
                                bollingerStdIndicatorName: newBollingerStdIndicatorName,
                                macdIndicatorName: newMACDIndicatorName,
                                macdHistogramIndicatorName: newMACDHistogramIndicatorName,
                                macdSignalIndicatorName: newMACDSignalIndicatorName}, axis=1)
        
        return algoDataDaily



    # Concatent weekly and daily indicators and clean dataset

    @st.cache(allow_output_mutation=True)
    def combineData(algoDataDaily, algoDataWeekly):
        algoData = pd.concat([algoDataDaily, algoDataWeekly], axis=1)
        algoData = algoData.interpolate(method='linear', limit_direction='forward', axis=0)
        algoData = algoData.dropna()
        
        return algoData
    
    
    @st.cache(allow_output_mutation=True)
    def convert_df(df):
        return df.to_csv().encode('utf-8')

    # Buy and Hold Strategy Returns

    @st.cache(allow_output_mutation=True)
    def buyHoldReturns(algoData):

        # Make first return and signal 0 so all cumulative returns start at 1
        algoData["ActualReturns"][0] = 0
        algoData["Signal"][0] = 0

        algoData['cumBuyHoldReturns'] = (1+algoData['ActualReturns']).cumprod()
        
        returns = algoData[['Signal', "ActualReturns", 'cumBuyHoldReturns']]
        
        return returns



    # MACD Strategy Function

    @st.cache(allow_output_mutation=True)
    def macdStrategy(algoData):
        
        signal = [0]
        
        for ind in range(1, algoData.shape[0]):
        
            if((algoData['MACDline'][ind] > algoData['MACDSignal'][ind]) & (algoData['MACDline'][ind-1] > algoData['MACDSignal'][ind-1]) & (algoData['weeklyEMAShort'][ind] > algoData['weeklyEMAShort'][ind-1]) & (algoData['weeklyEMALong'][ind] > algoData['weeklyEMALong'][ind-1]) & (algoData['weeklyADX'][ind] > 25)):
            
                signal.append(1)
        
            elif ((algoData['MACDline'][ind] < algoData['MACDSignal'][ind]) & (algoData['MACDline'][ind-1] < algoData['MACDSignal'][ind-1]) & (algoData['weeklyEMAShort'][ind] < algoData['weeklyEMAShort'][ind-1]) & (algoData['weeklyEMALong'][ind] < algoData['weeklyEMALong'][ind-1]) & (algoData['weeklyADX'][ind] > 25)):

                signal.append(-1)
        
            else:
            
                signal.append(0)
            
        return signal



    # MACD Strategy returns

    @st.cache(allow_output_mutation=True)
    def macdReturns(algoData):

        # Make first return 0 so all cumulative returns start at 1
        algoData["ActualReturns"][0] = 0

        algoData['MACDStrategy'] = macdStrategy(algoData)
        algoData['MACDStrategyReturns'] = algoData['ActualReturns'] * algoData['MACDStrategy'].shift()
        algoData['MACDStrategyReturns'] = algoData['MACDStrategyReturns'].fillna(0)
        algoData['cumMACDReturns'] = (1 + algoData['MACDStrategyReturns']).cumprod()
        
        returns = algoData[['MACDStrategy', 'MACDStrategyReturns', 'cumMACDReturns']]
        
        return returns



    # Elder-Ray Strategy Function

    @st.cache(allow_output_mutation=True)
    def elderRaySystem(algoData):
        
        signal = [0]
        
        for ind in range(1, algoData.shape[0]):
        
            if((algoData['dailyEMAShort'][ind] > algoData['dailyEMAShort'][ind-1]) & (algoData['dailyEMALong'][ind] > algoData['dailyEMALong'][ind-1]) & (algoData['BearPower'][ind] > algoData['BearPower'][ind-1]) & (algoData['weeklyEMAShort'][ind] > algoData['weeklyEMAShort'][ind-1]) & (algoData['weeklyEMALong'][ind] > algoData['weeklyEMALong'][ind-1]) & (algoData['weeklyADX'][ind] > 25)):
            
                signal.append(1)
        
            elif ((algoData['dailyEMAShort'][ind] < algoData['dailyEMAShort'][ind-1]) & (algoData['dailyEMALong'][ind] < algoData['dailyEMALong'][ind-1]) & (algoData['BullPower'][ind] < algoData['BullPower'][ind-1]) & (algoData['weeklyEMAShort'][ind] < algoData['weeklyEMAShort'][ind-1]) & (algoData['weeklyEMALong'][ind] < algoData['weeklyEMALong'][ind-1]) & (algoData['weeklyADX'][ind] > 25)):

                signal.append(-1)
        
            else:
            
                signal.append(0)
            
        return signal



    # Elder Ray Strategy returns

    @st.cache(allow_output_mutation=True)
    def elderRayReturns(algoData):

        # Make first return 0 so all cumulative returns start at 1
        algoData["ActualReturns"][0] = 0

        algoData['ElderRayStrategy'] = elderRaySystem(algoData)
        algoData['ElderRayStrategyReturns'] = algoData['ActualReturns'] * algoData['ElderRayStrategy'].shift()
        algoData['ElderRayStrategyReturns'] = algoData['ElderRayStrategyReturns'].fillna(0)
        algoData['cumElderRayReturns'] = (1 + algoData['ElderRayStrategyReturns']).cumprod()
        
        returns = algoData[['ElderRayStrategy', 'ElderRayStrategyReturns', 'cumElderRayReturns']]
        
        return returns



    # Impulse System Function

    @st.cache(allow_output_mutation=True)
    def impulseSystem(algoData):
        
        signal = [0]
        
        for ind in range(1, algoData.shape[0]):
        
            if((algoData['dailyEMAShort'][ind] > algoData['dailyEMAShort'][ind-1]) & (algoData['dailyEMALong'][ind] > algoData['dailyEMALong'][ind-1]) & (algoData['MACDHistogram'][ind] > algoData['MACDHistogram'][ind-1]) & (algoData['weeklyEMAShort'][ind] > algoData['weeklyEMAShort'][ind-1]) & (algoData['weeklyEMALong'][ind] > algoData['weeklyEMALong'][ind-1]) & (algoData['weeklyADX'][ind] > 25)):
            
                signal.append(1)
        
            elif ((algoData['dailyEMAShort'][ind] < algoData['dailyEMAShort'][ind-1]) & (algoData['dailyEMALong'][ind] < algoData['dailyEMALong'][ind-1]) & (algoData['MACDHistogram'][ind] < algoData['MACDHistogram'][ind-1]) & (algoData['weeklyEMAShort'][ind] < algoData['weeklyEMAShort'][ind-1]) & (algoData['weeklyEMALong'][ind] < algoData['weeklyEMALong'][ind-1]) & (algoData['weeklyADX'][ind] > 25)):

                signal.append(-1)
        
            else:
            
                signal.append(0)
            
        return signal


    # Impulse Strategy returns

    @st.cache(allow_output_mutation=True)
    def impulseReturns(algoData):

        # Make first return 0 so all cumulative returns start at 1
        algoData["ActualReturns"][0] = 0

        algoData['ImpulseStrategy'] = impulseSystem(algoData)
        algoData['ImpulseStrategyReturns'] = algoData['ActualReturns'] * algoData['ImpulseStrategy'].shift()
        algoData['ImpulseStrategyReturns'] = algoData['ImpulseStrategyReturns'].fillna(0)
        algoData['cumImpulseReturns'] = (1 + algoData['ImpulseStrategyReturns']).cumprod()
        
        returns = algoData[['ImpulseStrategy', 'ImpulseStrategyReturns', 'cumImpulseReturns']]
        
        return returns



    # Bollinger Bands Function

    @st.cache(allow_output_mutation=True)
    def bbStrategy(algoData):
        
        signal = [0]
        position = False
        
        for ind in range(1, algoData.shape[0]):
            
            if ((algoData['Close'][ind] > algoData['upperBB'][ind]) & (algoData['Close'][ind-1] < algoData['upperBB'][ind-1]) & (position == False)):
                signal.append(-1)
                position = True
            elif ((algoData['Close'][ind] < algoData['lowerBB'][ind]) & (algoData['Close'][ind-1] > algoData['lowerBB'][ind-1]) & (position == False)):
                signal.append(1)
                position = True
            elif ((algoData['Close'][ind] < algoData['middleBB'][ind]) & (algoData['Close'][ind-1] > algoData['middleBB'][ind-1]) & (position == True)):
                signal.append(0)
                position = False
            elif ((algoData['Close'][ind] > algoData['middleBB'][ind]) & (algoData['Close'][ind-1] < algoData['middleBB'][ind-1]) & (position == True)):
                signal.append(0)
                position = False
            else:
                signal.append(np.nan)
        
        return signal


    # Bollinger Bands Strategy returns

    @st.cache(allow_output_mutation=True)
    def bollingerReturns(algoData):

        # Make first return 0 so all cumulative returns start at 1
        algoData["ActualReturns"][0] = 0

        # Caluclate BB Strategy
        algoData['BBStrategy'] = bbStrategy(algoData)
        algoData['BBStrategy'] = algoData['BBStrategy'].ffill()
        algoData['BBStrategyReturns'] = algoData['ActualReturns'] * algoData['BBStrategy'].shift()
        algoData['BBStrategyReturns'] = algoData['BBStrategyReturns'].fillna(0)
        algoData['cumBBReturns'] = (1 + algoData['BBStrategyReturns']).cumprod()

        returns = algoData[['BBStrategy', 'BBStrategyReturns', 'cumBBReturns']]
        
        return returns


    # Calculate all Strategy Returns and place in a dataframe 

    @st.cache(allow_output_mutation=True)
    def allReturnsData(algoData):
        
        buyHoldReturn = buyHoldReturns(algoData)
        macdReturn = macdReturns(algoData)
        elderRayReturn = elderRayReturns(algoData)
        impulseReturn = impulseReturns(algoData)
        bollingerReturn = bollingerReturns(algoData)
        
        allReturns = pd.concat([buyHoldReturn, macdReturn, elderRayReturn, impulseReturn, bollingerReturn], axis=1)
        allReturns = allReturns.dropna()
        
        return allReturns



    # Plot Strategy Returns

    def cumulativeStrategyReturnsPlot(allStrategyReturns, assetName, period):

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=allStrategyReturns.index,
                y=allStrategyReturns['cumBuyHoldReturns'],
                name="Buy&Hold",
                line=dict(color="green")
            ))

        fig.add_trace(
            go.Scatter(
                x=allStrategyReturns.index,
                y=allStrategyReturns['cumMACDReturns'],
                name='MACD',
                line=dict(color="red")
            ))


        fig.add_trace(
            go.Scatter(
                x=allStrategyReturns.index,
                y=allStrategyReturns['cumElderRayReturns'],
                name='Elder-Ray',
                line=dict(color="blue")
            ))

        fig.add_trace(
            go.Scatter(
                x=allStrategyReturns.index,
                y=allStrategyReturns['cumImpulseReturns'],
                name='Impulse System',
                line=dict(color="orange")
            ))

        fig.add_trace(
            go.Scatter(
                x=allStrategyReturns.index,
                y=allStrategyReturns['cumBBReturns'],
                name='Bollinger Bands',
                line=dict(color="purple")
            ))



        fig.update_layout(
            title={
                'text': "Cumulative Strategy Returns",
            },
            template='seaborn',
            xaxis=dict(autorange=True,
                    title_text='Date'),
            yaxis=dict(autorange=True,
                    title_text='Cumulative Returns'),
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1,
                xanchor='right',
                x=1
            ))
        
        return fig




    # Descriptive Statistics Function

    def descriptiveStats(allStrategyReturns, initial, riskFree, assetName, period):

        # Calculate Descriptive Statistics

        start_date = allStrategyReturns.index.min()
        end_date = allStrategyReturns.index.max()

        start = str(start_date.day)+'-'+str(start_date.month)+'-'+str(start_date.year)
        end = str(end_date.day)+'-'+str(end_date.month)+'-'+str(end_date.year)

        days = (end_date - start_date).days
        years = days/365

        init_investment = initial
        rf = riskFree

        buyHold_start = init_investment
        macd_start = init_investment
        elderRay_start = init_investment
        impulse_start = init_investment
        bollinger_start = init_investment

        buyHold_end = round(allStrategyReturns['cumBuyHoldReturns'][-1] * init_investment,2)
        macd_end = round(allStrategyReturns['cumMACDReturns'][-1] * init_investment,2)
        elderRay_end = round(allStrategyReturns['cumElderRayReturns'][-1] * init_investment,2)
        impulse_end = round(allStrategyReturns['cumImpulseReturns'][-1] * init_investment,2)
        bollinger_end = round(allStrategyReturns['cumBBReturns'][-1] * init_investment,2)

        buyHold_max_dailyReturn = round(allStrategyReturns['ActualReturns'].max(),6)
        macd_max_dailyReturn = round(allStrategyReturns['MACDStrategyReturns'].max(),6)
        elderRay_max_dailyReturn = round(allStrategyReturns['ElderRayStrategyReturns'].max(),6)
        impulse_max_dailyReturn = round(allStrategyReturns['ImpulseStrategyReturns'].max(),6)
        bollinger_max_dailyReturn = round(allStrategyReturns['BBStrategyReturns'].max(),6)

        buyHold_min_dailyReturn = round(allStrategyReturns['ActualReturns'].min(),6)
        macd_min_dailyReturn = round(allStrategyReturns['MACDStrategyReturns'].min(),6)
        elderRay_min_dailyReturn = round(allStrategyReturns['ElderRayStrategyReturns'].min(),6)
        impulse_min_dailyReturn = round(allStrategyReturns['ImpulseStrategyReturns'].min(),6)
        bollinger_min_dailyReturn = round(allStrategyReturns['BBStrategyReturns'].min(),6)

        buyHold_max_drawdown = round(((allStrategyReturns['cumBuyHoldReturns'].min() - allStrategyReturns['cumBuyHoldReturns'].max())/allStrategyReturns['cumBuyHoldReturns'].max()),6)
        macd_max_drawdown = round(((allStrategyReturns['cumMACDReturns'].min() - allStrategyReturns['cumMACDReturns'].max())/allStrategyReturns['cumMACDReturns'].max()),6)
        elderRay_max_drawdown = round(((allStrategyReturns['cumElderRayReturns'].min() - allStrategyReturns['cumElderRayReturns'].max())/allStrategyReturns['cumElderRayReturns'].max()),6)
        impulse_max_drawdown = round(((allStrategyReturns['cumImpulseReturns'].min() - allStrategyReturns['cumImpulseReturns'].max())/allStrategyReturns['cumImpulseReturns'].max()),6)
        bollinger_max_drawdown = round(((allStrategyReturns['cumBBReturns'].min() - allStrategyReturns['cumBBReturns'].max())/allStrategyReturns['cumBBReturns'].max()),6)

        
        
        buyHoldSignals = allStrategyReturns.Signal[(allStrategyReturns['Signal'] == 1) | (allStrategyReturns['Signal'] == -1)].count()
        macdSignals = allStrategyReturns.MACDStrategy[(allStrategyReturns['MACDStrategy'] == 1) | (allStrategyReturns['MACDStrategy'] == -1)].count()
        elderRaySignals = allStrategyReturns.ElderRayStrategy[(allStrategyReturns['ElderRayStrategy'] == 1) | (allStrategyReturns['ElderRayStrategy'] == -1)].count()
        impulseSignals = allStrategyReturns.ImpulseStrategy[(allStrategyReturns['ImpulseStrategy'] == 1) | (allStrategyReturns['ImpulseStrategy'] == -1)].count()
        bollingerSignals = allStrategyReturns.BBStrategy[(allStrategyReturns['BBStrategy'] == 1) | (allStrategyReturns['BBStrategy'] == -1)].count()
        
    
        
        buyHoldPos = allStrategyReturns.ActualReturns[(allStrategyReturns['ActualReturns'] > 0)].count()
        macdPos = allStrategyReturns.MACDStrategyReturns[(allStrategyReturns['MACDStrategyReturns'] > 0)].count()
        elderRayPos = allStrategyReturns.ElderRayStrategyReturns[(allStrategyReturns['ElderRayStrategyReturns'] > 0)].count()
        impulsePos = allStrategyReturns.ImpulseStrategyReturns[(allStrategyReturns['ImpulseStrategyReturns'] > 0)].count()
        bollingerPos = allStrategyReturns.BBStrategyReturns[(allStrategyReturns['BBStrategyReturns'] > 0)].count()
        
        
        buyHoldPosPerc = round((buyHoldPos/buyHoldSignals),6)
        macdPosPerc = round((macdPos/macdSignals),6)
        elderRayPosPerc = round((elderRayPos/elderRaySignals),6)
        impulsePosPerc = round((impulsePos/impulseSignals),6)
        bollingerPosPerc = round((bollingerPos/bollingerSignals),6)
        
        
        buyHoldPosSum = allStrategyReturns.ActualReturns[(allStrategyReturns['ActualReturns'] > 0)].sum()
        macdPosSum = allStrategyReturns.MACDStrategyReturns[(allStrategyReturns['MACDStrategyReturns'] > 0)].sum()
        elderRayPosSum = allStrategyReturns.ElderRayStrategyReturns[(allStrategyReturns['ElderRayStrategyReturns'] > 0)].sum()
        impulsePosSum = allStrategyReturns.ImpulseStrategyReturns[(allStrategyReturns['ImpulseStrategyReturns'] > 0)].sum()
        bollingerPosSum = allStrategyReturns.BBStrategyReturns[(allStrategyReturns['BBStrategyReturns'] > 0)].sum()
        
        buyHoldPosAvg = round((buyHoldPosSum/buyHoldPos),6)
        macdPosAvg = round((macdPosSum/macdPos),6)
        elderRayPosAvg = round((elderRayPosSum/elderRayPos),6)
        impulsePosAvg = round((impulsePosSum/impulsePos),6)
        bollingerPosAvg = round((bollingerPosSum/bollingerPos),6)
        
        
        
        buyHoldNeg = allStrategyReturns.ActualReturns[(allStrategyReturns['ActualReturns'] < 0)].count()
        macdNeg = allStrategyReturns.MACDStrategyReturns[(allStrategyReturns['MACDStrategyReturns'] < 0)].count()
        elderRayNeg = allStrategyReturns.ElderRayStrategyReturns[(allStrategyReturns['ElderRayStrategyReturns'] < 0)].count()
        impulseNeg = allStrategyReturns.ImpulseStrategyReturns[(allStrategyReturns['ImpulseStrategyReturns'] < 0)].count()
        bollingerNeg = allStrategyReturns.BBStrategyReturns[(allStrategyReturns['BBStrategyReturns'] < 0)].count()
        
        
        buyHoldNegPerc = round((buyHoldNeg/buyHoldSignals),6)
        macdNegPerc = round((macdNeg/macdSignals),6)
        elderRayNegPerc = round((elderRayNeg/elderRaySignals),6)
        impulseNegPerc = round((impulseNeg/impulseSignals),6)
        bollingerNegPerc = round((bollingerNeg/bollingerSignals),6)
        
        
        
        buyHoldNegSum = allStrategyReturns.ActualReturns[(allStrategyReturns['ActualReturns'] < 0)].sum()
        macdNegSum = allStrategyReturns.MACDStrategyReturns[(allStrategyReturns['MACDStrategyReturns'] < 0)].sum()
        elderRayNegSum = allStrategyReturns.ElderRayStrategyReturns[(allStrategyReturns['ElderRayStrategyReturns'] < 0)].sum()
        impulseNegSum = allStrategyReturns.ImpulseStrategyReturns[(allStrategyReturns['ImpulseStrategyReturns'] < 0)].sum()
        bollingerNegSum = allStrategyReturns.BBStrategyReturns[(allStrategyReturns['BBStrategyReturns'] < 0)].sum()
        
        buyHoldNegAvg = round((buyHoldNegSum/buyHoldNeg),6)
        macdNegAvg = round((macdNegSum/macdNeg),6)
        elderRayNegAvg = round((elderRayNegSum/elderRayNeg),6)
        impulseNegAvg = round((impulseNegSum/impulseNeg),6)
        bollingerNegAvg = round((bollingerNegSum/bollingerNeg),6)
        
        
        
        buyHold_annualReturn = allStrategyReturns['ActualReturns'].apply(lambda x: (1+x)).cumprod().iloc[-1]**(1/years) - 1
        macd_annualReturn = allStrategyReturns['MACDStrategyReturns'].apply(lambda x: (1+x)).cumprod().iloc[-1]**(1/years) - 1
        elderRay_annualReturn = allStrategyReturns['ElderRayStrategyReturns'].apply(lambda x: (1+x)).cumprod().iloc[-1]**(1/years) - 1
        impulse_annualReturn = allStrategyReturns['ImpulseStrategyReturns'].apply(lambda x: (1+x)).cumprod().iloc[-1]**(1/years) - 1
        bollinger_annualReturn = allStrategyReturns['BBStrategyReturns'].apply(lambda x: (1+x)).cumprod().iloc[-1]**(1/years) - 1


        buyHold_annualVol = allStrategyReturns['ActualReturns'].apply(lambda x: np.log(1+x)).std()*np.sqrt(252)
        macd_annualVol = allStrategyReturns['MACDStrategyReturns'].apply(lambda x: np.log(1+x)).std()*np.sqrt(252)
        elderRay_annualVol = allStrategyReturns['ElderRayStrategyReturns'].apply(lambda x: np.log(1+x)).std()*np.sqrt(252)
        impulse_annualVol = allStrategyReturns['ImpulseStrategyReturns'].apply(lambda x: np.log(1+x)).std()*np.sqrt(252)
        bollinger_annualVol = allStrategyReturns['BBStrategyReturns'].apply(lambda x: np.log(1+x)).std()*np.sqrt(252)

        buyHold_Sharpe = round((buyHold_annualReturn-rf)/buyHold_annualVol,2)
        macd_Sharpe = round((macd_annualReturn-rf)/macd_annualVol,2)
        elderRay_Sharpe = round((elderRay_annualReturn-rf)/elderRay_annualVol,2)
        impulse_Sharpe = round((impulse_annualReturn-rf)/impulse_annualVol,2)
        bollinger_Sharpe = round((bollinger_annualReturn-rf)/bollinger_annualVol,2)


        buyHold_variance = allStrategyReturns['ActualReturns'].var()

        macd_covariance = allStrategyReturns['MACDStrategyReturns'].cov(allStrategyReturns['ActualReturns'])
        elderRay_covariance = allStrategyReturns['ElderRayStrategyReturns'].cov(allStrategyReturns['ActualReturns'])
        impulse_covariance = allStrategyReturns['ImpulseStrategyReturns'].cov(allStrategyReturns['ActualReturns'])
        bollinger_covariance = allStrategyReturns['BBStrategyReturns'].cov(allStrategyReturns['ActualReturns'])

        buyHold_beta = 1

        macd_beta = round(macd_covariance/buyHold_variance,2)
        elderRay_beta = round(elderRay_covariance/buyHold_variance,2)
        impulse_beta = round(impulse_covariance/buyHold_variance,2)
        bollinger_beta = round(bollinger_covariance/buyHold_variance,2)


        # Table of Descriptive Statistics

        head = ['<b>Statistic<b>', '<b>Buy&Hold<b>', '<b>MACD<b>', '<b>Elder-Ray<b>', '<b>Impulse<b>', '<b>Bollinger<b>']
        labels = ['<b>Start Date<b>', '<b>End Date<b>','<b>Initial Investment<b>', '<b>Ending Investment<b>','--------------------------',
                '<b>Signals<b>', '<b>Winning Trades<b>', '<b>Losing Trades<b>', '<b>% Winning<b>', '<b>% Losing<b>', 
                '<b>Average Profit<b>','<b>Average Loss<b>', '--------------------------', 
                '<b>Max Daily Return<b>','<b>Min Daily Return<b>', '<b>Max Drawdown<b>', '<b>Annual Return<b>',
                '<b>Annual Volatility<b>', '<b>Sharpe Ratio<b>', '<b>Beta<b>']


        buyHold_stats = [start, end,'${:,}'.format(buyHold_start), '${:,}'.format(buyHold_end), '--------------------------',
                        buyHoldSignals, buyHoldPos, buyHoldNeg,'{:.2%}'.format(buyHoldPosPerc),'{:.2%}'.format(buyHoldNegPerc), 
                        '{:.2%}'.format(buyHoldPosAvg), '{:.2%}'.format(buyHoldNegAvg), '--------------------------', 
                        '{:.2%}'.format(buyHold_max_dailyReturn), '{:.2%}'.format(buyHold_min_dailyReturn), 
                        '{:.2%}'.format(buyHold_max_drawdown), '{:.2%}'.format(buyHold_annualReturn), 
                        '{:.2%}'.format(buyHold_annualVol), buyHold_Sharpe, buyHold_beta]


        macd_stats = [start, end,'${:,}'.format(macd_start), '${:,}'.format(macd_end),'--------------------------', 
                    macdSignals, macdPos, macdNeg,'{:.2%}'.format(macdPosPerc),'{:.2%}'.format(macdNegPerc),
                    '{:.2%}'.format(macdPosAvg), '{:.2%}'.format(macdNegAvg), '--------------------------', 
                    '{:.2%}'.format(macd_max_dailyReturn), '{:.2%}'.format(macd_min_dailyReturn), 
                    '{:.2%}'.format(macd_max_drawdown), '{:.2%}'.format(macd_annualReturn), 
                    '{:.2%}'.format(macd_annualVol), macd_Sharpe, macd_beta]

        elderRay_stats = [start, end, '${:,}'.format(elderRay_start), '${:,}'.format(elderRay_end),'--------------------------', 
                        elderRaySignals, elderRayPos, elderRayNeg, '{:.2%}'.format(elderRayPosPerc),
                        '{:.2%}'.format(elderRayNegPerc), '{:.2%}'.format(elderRayPosAvg), '{:.2%}'.format(elderRayNegAvg), 
                        '--------------------------', '{:.2%}'.format(elderRay_max_dailyReturn), 
                        '{:.2%}'.format(elderRay_min_dailyReturn), '{:.2%}'.format(elderRay_max_drawdown),
                        '{:.2%}'.format(elderRay_annualReturn), '{:.2%}'.format(elderRay_annualVol), 
                        elderRay_Sharpe, elderRay_beta]


        impulse_stats = [start, end, '${:,}'.format(impulse_start), '${:,}'.format(impulse_end), '--------------------------', 
                        impulseSignals, impulsePos, impulseNeg, '{:.2%}'.format(impulsePosPerc),'{:.2%}'.format(impulseNegPerc), 
                        '{:.2%}'.format(impulsePosAvg), '{:.2%}'.format(impulseNegAvg), '--------------------------', 
                        '{:.2%}'.format(impulse_max_dailyReturn), '{:.2%}'.format(impulse_min_dailyReturn), 
                        '{:.2%}'.format(impulse_max_drawdown), '{:.2%}'.format(impulse_annualReturn), 
                        '{:.2%}'.format(impulse_annualVol), impulse_Sharpe, impulse_beta]


        bollinger_stats = [start, end, '${:,}'.format(bollinger_start), '${:,}'.format(bollinger_end), '--------------------------',
                        bollingerSignals, bollingerPos, bollingerNeg,'{:.2%}'.format(bollingerPosPerc),
                        '{:.2%}'.format(bollingerNegPerc), '{:.2%}'.format(bollingerPosAvg), 
                        '{:.2%}'.format(bollingerNegAvg), '--------------------------', '{:.2%}'.format(bollinger_max_dailyReturn), 
                        '{:.2%}'.format(bollinger_min_dailyReturn), '{:.2%}'.format(bollinger_max_drawdown), 
                        '{:.2%}'.format(bollinger_annualReturn), '{:.2%}'.format(bollinger_annualVol), 
                        bollinger_Sharpe, bollinger_beta]



        fig11 = go.Figure(data=[go.Table(
            header=dict(values=head,
                    fill_color='paleturquoise',
                    align='left'),
            cells=dict(values=[labels, buyHold_stats, macd_stats, elderRay_stats, impulse_stats, bollinger_stats],
                fill_color='lavender',
                align='left'))
        ])

        fig11.update_layout(margin=dict(l=0, r=0, b=0,t=0), width=750, height=650)
        
        return fig11


    ##############################################################################################################################################
    ################################################ Generate Output #############################################################################

    # Get Technical Analysis and Signal Data for all Assets

    algoDataWeekly = weeklyIndicators(assetCodes, timePeriod, daily, start_date, end_date, weeklyEMAShort, weeklyEMALong, weeklyADX, 
                                    weeklyEMAShortIndicatorName, weeklyEMALongIndicatorName, weeklyADXName, newWeeklyEMAShortIndicatorName,
                                    newWeeklyEMALongIndicatorName, newWeeklyADXName)
    
    algoDataWeeklyDf= convert_df(algoDataWeekly)


    algoDataDaily = dailyIndicators(assetCodes, timePeriod, daily, start_date, end_date, dailyEMAShort, dailyEMALong, elderRayLength,
                                    bbandsLength, macdFast, macdSlow, macdSignal, dailyEMAShortIndicatorName,
                                    dailyEMALongIndicatorName,bullPowerIndicatorName, bearPowerIndicatorName,
                                    bollingerLowerIndicatorName, bollingerMiddleIndicatorName, bollingerUpperIndicatorName,
                                    bollingerStdIndicatorName, macdIndicatorName,macdHistogramIndicatorName,
                                    macdSignalIndicatorName, newDailylyEMAShortIndicatorName,newDailyEMALongIndicatorName,
                                    newBullPowerIndicatorName, newBearPowerIndicatorName, newBollingerLowerIndicatorName,
                                    newBollingerMiddleIndicatorName, newBollingerUpperIndicatorName,newBollingerStdIndicatorName,
                                    newMACDIndicatorName, newMACDHistogramIndicatorName,newMACDSignalIndicatorName)
    
    algoDataDailyDf = convert_df(algoDataDaily)


    algoData = combineData(algoDataDaily, algoDataWeekly)
    algoDataDf = convert_df(algoData)

  
    # Split the data into training and test datasets

    # Assign a copy of the aldoData DataFrame
    returns = algoData[algoData.columns]

    # Select the start of the training period
    returns_begin = returns.index.min()

    # Select the ending period for the training data with an offset of x months
    returns_end = returns.index.min() + DateOffset(months=training_months)

    # Generate the taining DataFrames
    returns_train = returns.loc[returns_begin:returns_end]

    # Generate the test DataFrames
    returns_test = returns.loc[returns_end+DateOffset(hours=1):]
        
    # Calulate all the Strategy returns for the training period
    allStrategyReturnsTrain = allReturnsData(returns_train)
    allStrategyReturnsTrainDf = convert_df(allStrategyReturnsTrain)


    # Plot the all the Strategy returns for the training period
    cumulativeStrategyReturnsPlotTrainingPeriod = cumulativeStrategyReturnsPlot(allStrategyReturnsTrain, assetNames, 'training')
    
    
    # Descriptive Statistics for all Strategy returns for the training period
    descriptiveStatsTrainingPeriod = descriptiveStats(allStrategyReturnsTrain, initial, riskFree, assetNames, 'training')
    
          
    # Calulate all the Strategy returns for the test period
    allStrategyReturnsTest = allReturnsData(returns_test)
    allStrategyReturnsTestDf = convert_df(allStrategyReturnsTest)


    # Plot the all the Strategy returns for the test period
    cumulativeStrategyReturnsPlotTestPeriod = cumulativeStrategyReturnsPlot(allStrategyReturnsTest, assetNames, 'test')
    
    # Descriptive Statistics for all Strategy returns for the test period
    descriptiveStatsTestPeriod = descriptiveStats(allStrategyReturnsTest, initial, riskFree, assetNames, 'test')
    
    
    ##############################################################################################################################################
    ################################################ Generate Graphs and Tables ##################################################################
    
    st.markdown("<h3 style='text-align: left; color: #872657; padding-left: 0px; font-size: 30px'><b>Data Download<b></h3>", unsafe_allow_html=True)
    
    colg, colh, coli, colj = st.columns([1, 1, 1, 1])
    
    colg.download_button(label="Daily Indicators", data=algoDataDailyDf, file_name=assetNames+'_dailyIndicators.csv', mime='text/csv')
    colh.download_button(label="Weekly Indicators", data=algoDataWeeklyDf, file_name=assetNames+'_weeklyIndicators.csv', mime='text/csv')
    coli.download_button(label="Merged Indicators", data=algoDataDf, file_name=assetNames+'_mergedIndicators.csv', mime='text/csv')
    colj.download_button(label="Test Period Signals", data=allStrategyReturnsTestDf, file_name=assetNames+'_tesReturns.csv', mime='text/csv')
    
    st.markdown(' ')
    st.markdown(' ')
    st.markdown("<h4 style='text-align: left; color: #551A8B; padding-left: 0px; font-size: 40px'><b>"+assetNames+"-Training Period<b></h4>", unsafe_allow_html=True)
    st.markdown(' ')
       
    st.plotly_chart(cumulativeStrategyReturnsPlotTrainingPeriod)
    st.plotly_chart(descriptiveStatsTrainingPeriod)
    
    st.markdown(' ')
    st.markdown(' ')
    st.markdown("<h4 style='text-align: left; color: #551A8B; padding-left: 0px; font-size: 40px'><b>"+assetNames+"-Test Period<b></h4>", unsafe_allow_html=True)
    st.markdown(' ')
    
    st.plotly_chart(cumulativeStrategyReturnsPlotTestPeriod)
    st.plotly_chart(descriptiveStatsTestPeriod)
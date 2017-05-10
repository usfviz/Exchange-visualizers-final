library(ggvis)

shinyUI(navbarPage("Currency Run", 
                   tabPanel("Performance",
                            sidebarLayout(
                              sidebarPanel(
                                textInput("ticker", "Stock Ticker", value = "AAPL"),
                                sliderInput('zoom', 'Zoom', min = 10, max=100, value=50),
                                checkboxInput('stream', 'Stream', value=T),
                                actionButton('scrollBack', "Step Back"),
                                actionButton('scrollForward', "Step Forward")
                                
                              ),
                              
                              #Render the results
                              mainPanel(
                                fluidRow(
                                  plotOutput("plot",
                                             click = "plot_click",
                                             dblclick = "plot_dblclick",
                                             hover = "plot_hover",
                                             brush = "plot_brush")
                                ),
                                fluidRow(verbatimTextOutput("summary"),
                                         verbatimTextOutput('info'))
                              )
                            )
                            
                   ),
                   tabPanel('Stock Data',
                            dataTableOutput('rawData')),
                   tabPanel('Trades',
                            dataTableOutput('tradeData')),
                   tabPanel('About',
                            "This dashboard was developed to visualize the behavior of an automated trading strategy on
                            real stock data. \nIt serves as the front end to a python script which collects the data, simulates
                            trades on the data and then serves the results back to R. This simulation assumes that you have a wallet
                            with a single dollar in it, and every time you trade your dollar increases or decreases in value.
                            On the visualization, there are two types of trades being plotted: long and short. A long trade is 
                            when you buy the asset, and a short is when you buy against the asset and so profit when the price 
                            declines. The thicker the line, the larger percentage of your wallet you bet on the trade. Ideally,
                            you want to see many green lines going upwards, and many red lines going downward. The individual 
                            trades and the gain/loss observed on those trades can be seen in the `Trades` tab, where Bet Size 
                            refers to the fraction of your dollar.")
          )
        )

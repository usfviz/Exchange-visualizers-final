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
                            h5("Designed by Nick Levitt and Sam Haaf."),
                            h3("\n\nPurpose:"),
                            h5("\tThis dashboard was developed to visualize the behavior of an automated trading strategy on
                            real stock data."),
                            h3("Method:"),
                            h5("\tIt serves as the front end to a python script which collects the data, simulates
                            trades on the data and then serves the results back to R. This simulation assumes that you have a wallet
                            with a single dollar in it, with every trade your dollar increases or decreases in value."),
                            h3("Interpretation:"),
                            h5("\tOn the visualization, there are two types of trades being plotted: long and short. A long trade is 
                            when you buy the asset, and a short is when you buy against the asset and so profit when the price 
                            declines. The thicker the line, the larger percentage of your wallet you bet on the trade. Ideally,
                            you want to see many buying lines going upwards, and many selling lines going downward. The individual 
                            trades and the gain/loss observed on those trades can be seen in the `Trades` tab, where Bet Size 
                            refers to the fraction of your dollar.")
          )
        )
)

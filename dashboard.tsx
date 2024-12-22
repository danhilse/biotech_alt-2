const MarketDashboard = () => {
    const [marketData, setMarketData] = useState([]);
    const [selectedStock, setSelectedStock] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [lastUpdated, setLastUpdated] = useState(null);
  
    const fetchData = async () => {
      setLoading(true);
      setError(null);
      try {
        const response = await fetch('/market_data.json');
        if (!response.ok) {
          throw new Error('Failed to fetch market data');
        }
        const data = await response.json();
        setMarketData(data);
        setLastUpdated(new Date(data[0]?.timestamp || Date.now()));
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };
  
    useEffect(() => {
      fetchData();
      const interval = setInterval(fetchData, 5 * 60 * 1000);
      return () => clearInterval(interval);
    }, []);
  
    const getColor = (stock) => {
      const { priceChange, volumeChange, volumeVsAvg } = stock;
      if (priceChange > 5 && (volumeChange > 20 || volumeVsAvg > 50)) return '#22c55e';
      if (priceChange < -5 && (volumeChange > 20 || volumeVsAvg > 50)) return '#ef4444';
      if (Math.abs(priceChange) > 5 || volumeChange > 20 || volumeVsAvg > 50) return '#eab308';
      return '#6b7280';
    };
  
    const formatNumber = (num) => {
      if (!num) return 'N/A';
      if (num >= 1e9) return `${(num / 1e9).toFixed(2)}B`;
      if (num >= 1e6) return `${(num / 1e6).toFixed(2)}M`;
      if (num >= 1e3) return `${(num / 1e3).toFixed(2)}K`;
      return num.toFixed(2);
    };
  
    const CustomTooltip = ({ active, payload }) => {
      if (active && payload && payload.length) {
        const data = payload[0].payload;
        return (
          <div className="bg-white p-4 rounded-lg shadow-lg border">
            <p className="font-bold">{data.symbol}</p>
            <p className="text-sm">Price: ${data.price}</p>
            <p className="text-sm">Price Change: {data.priceChange}%</p>
            <p className="text-sm">Volume: {formatNumber(data.volume)}</p>
            <p className="text-sm">Volume Change: {data.volumeChange}%</p>
            <p className="text-sm">vs Avg Volume: {data.volumeVsAvg}%</p>
            <p className="text-sm">Market Cap: ${formatNumber(data.marketCap)}</p>
            <p className="text-sm">Alerts: {data.alerts}</p>
          </div>
        );
      }
      return null;
    };
  
    const VolumeChart = ({ data }) => {
      if (!data?.recentVolumes) return null;
      
      const chartData = data.recentVolumes.map((volume, index) => ({
        day: index === 4 ? 'Today' : `Day ${4-index}`,
        volume
      }));
  
      return (
        <div className="h-40">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="day" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="volume" fill="#3b82f6" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      );
    };
  
    const AlertIndicator = ({ condition, children }) => {
      if (!condition) return null;
      return (
        <div className="flex items-center gap-1 text-yellow-600">
          <AlertTriangle className="w-4 h-4" />
          {children}
        </div>
      );
    };
  
    const MetricCard = ({ title, value, change, changeLabel = "vs prev" }) => (
      <div className="p-4 bg-gray-50 rounded-lg">
        <p className="text-sm font-medium text-gray-600">{title}</p>
        <p className="text-lg font-semibold">{value}</p>
        {change != null && (
          <p className={`text-sm ${change > 0 ? 'text-green-600' : change < 0 ? 'text-red-600' : 'text-gray-600'}`}>
            {change > 0 ? '↑' : change < 0 ? '↓' : '→'} {Math.abs(change).toFixed(2)}% {changeLabel}
          </p>
        )}
      </div>
    );
  
    if (loading && !marketData.length) {
      return (
        <div className="w-full h-96 flex items-center justify-center">
          <RefreshCw className="w-6 h-6 animate-spin mr-2" />
          <p>Loading market data...</p>
        </div>
      );
    }
  
    return (
      <div className="w-full space-y-4">
        <div className="flex justify-between items-center">
          <h2 className="text-2xl font-bold">Biotech Market Overview</h2>
          <div className="text-sm text-gray-500">
            {lastUpdated && `Last updated: ${lastUpdated.toLocaleTimeString()}`}
            <button 
              onClick={fetchData}
              className="ml-2 p-1 hover:bg-gray-100 rounded"
            >
              <RefreshCw className="w-4 h-4" />
            </button>
          </div>
        </div>
  
        {error && (
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}
  
        <Card>
          <CardContent className="pt-6">
            <div className="h-96">
              <ResponsiveContainer width="100%" height="100%">
                <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis
                    type="number"
                    dataKey="priceChange"
                    name="Price Change %"
                    domain={[-10, 10]}
                    label={{ value: 'Price Change %', position: 'bottom' }}
                  />
                  <YAxis
                    type="number"
                    dataKey="volumeVsAvg"
                    name="Volume vs 90d Avg %"
                    domain={[-50, 100]}
                    label={{ value: 'Volume vs Avg %', angle: -90, position: 'left' }}
                  />
                  <ZAxis
                    type="number"
                    dataKey="marketCap"
                    range={[50, 400]}
                    name="Market Cap"
                  />
                  <Tooltip content={<CustomTooltip />} />
                  <Scatter
                    data={marketData}
                    onClick={(data) => setSelectedStock(data)}
                  >
                    {marketData.map((entry, index) => (
                      <cell
                        key={index}
                        fill={getColor(entry)}
                      />
                    ))}
                  </Scatter>
                </ScatterChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
  
        {selectedStock && (
          <div className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center justify-between">
                  <span>{selectedStock.symbol} Overview</span>
                  {selectedStock.alerts > 0 && (
                    <span className="text-sm bg-yellow-100 text-yellow-800 px-2 py-1 rounded-full">
                      {selectedStock.alerts} Active Alerts
                    </span>
                  )}
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <MetricCard
                    title="Current Price"
                    value={`$${selectedStock.price}`}
                    change={selectedStock.priceChange}
                  />
                  <MetricCard
                    title="Trading Range"
                    value={`$${selectedStock.dayLow} - $${selectedStock.dayHigh}`}
                  />
                  <MetricCard
                    title="Volume"
                    value={formatNumber(selectedStock.volume)}
                    change={selectedStock.volumeChange}
                  />
                  <MetricCard
                    title="Market Cap"
                    value={`$${formatNumber(selectedStock.marketCap)}`}
                  />
                </div>
              </CardContent>
            </Card>
  
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <Card>
                <CardHeader>
                  <CardTitle>Volume Analysis</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div className="h-40">
                      <VolumeChart data={selectedStock} />
                    </div>
                    <div className="grid grid-cols-2 gap-4">
                      <MetricCard
                        title="vs 90d Average"
                        value={formatNumber(selectedStock.averageVolume)}
                        change={selectedStock.volumeVsAvg}
                        changeLabel="difference"
                      />
                      <MetricCard
                        title="Daily Change"
                        value={formatNumber(selectedStock.volume)}
                        change={selectedStock.volumeChange}
                      />
                    </div>
                  </div>
                </CardContent>
              </Card>
  
              <Card>
                <CardHeader>
                  <CardTitle>Alerts & Indicators</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2">
                    <AlertIndicator condition={selectedStock.alertDetails.priceAlert}>
                      Major price movement ({selectedStock.priceChange}%)
                    </AlertIndicator>
                    <AlertIndicator condition={selectedStock.alertDetails.volumeSpike20}>
                      Major volume spike ({selectedStock.volumeChange}% daily change)
                    </AlertIndicator>
                    <AlertIndicator condition={selectedStock.alertDetails.volumeSpike10}>
                      Notable volume increase
                    </AlertIndicator>
                    <AlertIndicator condition={selectedStock.alertDetails.highVolume}>
                      Significantly above average volume ({selectedStock.volumeVsAvg}% vs avg)
                    </AlertIndicator>
                  </div>
                  <div className="mt-4 grid grid-cols-2 gap-4">
                    <div className="p-4 bg-gray-50 rounded-lg">
                      <p className="text-sm font-medium text-gray-600">52-Week High</p>
                      <p className="text-lg font-semibold">
                        ${selectedStock.fiftyTwoWeekHigh || 'N/A'}
                      </p>
                    </div>
                    <div className="p-4 bg-gray-50 rounded-lg">
                      <p className="text-sm font-medium text-gray-600">52-Week Low</p>
                      <p className="text-lg font-semibold">
                        ${selectedStock.fiftyTwoWeekLow || 'N/A'}
                      </p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </div>
        )}
      </div>
    );
  };
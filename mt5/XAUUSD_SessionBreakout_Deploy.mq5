//+------------------------------------------------------------------+
//|                             XAUUSD_SessionBreakout_Deploy.mq5    |
//|                                                                  |
//|  Session opening-range breakout, deployment build.               |
//|                                                                  |
//|  Separate from both XAUUSD_SessionBreakout.mq5 (the 2024-2025    |
//|  original) and XAUUSD_SessionBreakout_2026.mq5 (the 2026 tune).  |
//|  Neither of those is touched; run this one alongside them for    |
//|  A/B if you want.                                                |
//|                                                                  |
//|  ================================================================|
//|  READ THIS BEFORE YOU TRADE IT                                   |
//|  ================================================================|
//|  The entry signal has NO measured directional edge. Forward      |
//|  return from the actual fill price over a fixed horizon, on      |
//|  696M ticks of Exness data:                                      |
//|                                                                  |
//|      2020-2023   +0.008 R    t = 0.68                            |
//|      2024        +0.052 R    t = 2.32                            |
//|      2025        +0.004 R    t = 0.17                            |
//|      2026        +0.011 R    t = 0.43   <- the year it was tuned |
//|                                                                  |
//|  So the "wider ranges mean more follow-through" premise in the   |
//|  2026 build is not supported. What this build changes is COST    |
//|  and PAYOFF SHAPE, both of which are real and measurable, and    |
//|  neither of which is an edge. Treat it as a long-volatility      |
//|  structure that is cheap to run in a high-volatility regime,     |
//|  and size it as a regime bet.                                    |
//|                                                                  |
//|  It takes roughly 3.3 years of live trading for this to reach    |
//|  t = 2. A 25-month underwater stretch is inside normal           |
//|  behaviour. If that horizon is not acceptable, do not deploy.    |
//|                                                                  |
//|  ================================================================|
//|  WHAT CHANGED FROM THE 2026 BUILD, AND WHY                       |
//|  ================================================================|
//|  Measured on the full 2020-2026 corpus at tick resolution, per   |
//|  unit of risk taken (1R = the bracket width = the stop):         |
//|                                                                  |
//|                        2026 build        this build              |
//|      dev 2020-2023     -0.056 R  t-2.15  +0.027 R  t+0.68        |
//|      val 2024-25H1     +0.080 R  t 1.95  +0.103 R  t 1.80        |
//|      2025H2-2026       +0.093 R  t 2.05  +0.131 R  t 2.31        |
//|      full corpus       +0.004 R  t 0.19  +0.079 R  t 2.74        |
//|      worst drawdown      444 R             62 R                  |
//|      trades / year       1,656             713                   |
//|                                                                  |
//|  1. InpMinRangePct 0.05 -> 0.25.                                 |
//|     THE MOST IMPORTANT CHANGE, and the only one justified in     |
//|     advance rather than by its own backtest. Cost is fixed in    |
//|     dollars, so it is a far bigger drag on a narrow bracket:     |
//|     0.117 R on the narrowest quintile against 0.027 R on the     |
//|     widest. Filtering them out improves BOTH periods, which a    |
//|     fitted parameter generally does not. It is a plateau, not a  |
//|     spike - 0.20, 0.25 and 0.30 all give roughly the same        |
//|     answer - so it is not a knife edge. It costs two thirds of   |
//|     the trades.                                                  |
//|                                                                  |
//|  2. Target 3x -> 4x width (PRESET_DEPLOY).                       |
//|     Weaker evidence: 4x had the best t-statistic of any target   |
//|     in 2024-2026 (2.99) and also improved dev, but it was        |
//|     chosen by looking. The direction is sound - every target     |
//|     from 3x upward beat 3x in both periods, and removing the     |
//|     target entirely was better still on expectancy. 4x is the    |
//|     compromise: uncapped earns more but 68% of its gain sits in  |
//|     25 trades out of 10,906, which is not something you can     |
//|     size on.                                                     |
//|                                                                  |
//|  3. Position size is now derived from the stop.                  |
//|     The stop IS the bracket width and the width varies 14x       |
//|     between its 5th and 95th percentile, so a flat lot size      |
//|     meant risk per trade varied 14x. Fixed lots also quietly     |
//|     made the account long volatility twice over, because wide    |
//|     brackets are exactly the high-volatility regime. Sizing to   |
//|     the stop LOWERS the backtested Sharpe (1.04 -> 0.07 on the   |
//|     full corpus) precisely because it removes that accidental    |
//|     bet. That is the point.                                      |
//|                                                                  |
//|  4. Volatility targeting is OFF by default.                      |
//|     With risk sizing on it is redundant and double-counts: the   |
//|     bracket width already scales with volatility, so sizing      |
//|     inversely to the width IS volatility targeting, done per     |
//|     trade instead of per day.                                    |
//|                                                                  |
//|  5. The daily loss halt is ON by default. It was 0.0 - off - in  |
//|     a system that can hold seven correlated positions.           |
//|                                                                  |
//|  6. Bug fixes carried in: ResolveOCO() and CancelWindowPending() |
//|     now filter orders by symbol as the other two loops already   |
//|     did; a missing M1 history no longer disables a window for    |
//|     the whole day; dead code removed.                            |
//|                                                                  |
//|  NOT changed: the stop stays on the opposite bracket edge.       |
//|  Anchoring it to the fill instead was tested and is identical    |
//|  to three decimal places, so it is not worth the code.           |
//|                                                                  |
//|  ---------------------------------------------------------------|
//|  SET InpGMTOffsetHours FIRST. Windows are defined in UTC. On     |
//|  Exness servers this is 0 and holds year-round (verified against |
//|  tick data across both daylight-saving transitions). On any      |
//|  other broker, check it - a wrong offset silently trades a       |
//|  completely different strategy.                                  |
//|  ---------------------------------------------------------------|
//+------------------------------------------------------------------+
#property copyright "Session breakout portfolio - deployment build"
#property version   "3.00"

#include <Trade\Trade.mqh>
#include <Trade\SymbolInfo.mqh>

#define MAX_WINDOWS 8

enum ENUM_PRESET
{
   PRESET_DEPLOY,           // 7 windows, 60m bracket, 4x target (recommended)
   PRESET_GEO_2026_NO_H13,  // the 2026 build's default, 3x target
   PRESET_GEO_2026,         // the 2026 build, 8 windows
   PRESET_TOP8_2026,        // 2026 hour re-selection (not supported by the data)
   PRESET_ORIGINAL          // the 2024-2025 configuration
};

//+------------------------------------------------------------------+
//| Inputs                                                           |
//+------------------------------------------------------------------+
input group "=== Broker clock (SET THIS FIRST) ==="
input int          InpGMTOffsetHours   = 0;      // Server time = GMT + this many hours
input bool         InpAutoDetectGMT    = true;   // Warn if the terminal disagrees

input group "=== Window set ==="
input ENUM_PRESET  InpPreset           = PRESET_DEPLOY;  // Which portfolio to trade
input double       InpTargetMultOverride = 0.0;  // Override every window's target (0 = use the preset)

input group "=== Position sizing ==="
input bool         InpUseRiskSizing    = true;   // Size from the stop distance (strongly recommended)
input double       InpRiskPctPerTrade  = 0.25;   // Risk per trade, % of equity, if sizing from the stop
input double       InpBaseLots         = 0.02;   // Fixed lots, used only when risk sizing is off
input bool         InpUseVolTargeting  = false;  // Redundant with risk sizing - see header note 4
input double       InpTargetVolPct     = 0.816;  // Reference daily realised vol (%)
input int          InpVolLookbackDays  = 20;     // Sessions in the trailing vol average
input double       InpMaxVolScale      = 3.0;    // Cap on the multiplier (floor is 1/x)

input group "=== Risk guards ==="
input double       InpMaxSpreadUSD     = 0.30;   // Skip entries above this spread, USD/oz (0 = off)
input int          InpMaxOpenPositions = 7;      // Portfolio-wide cap. 7 = as tested; see the warning at init
input double       InpMaxDailyLossPct  = 3.0;    // Halt for the day past this % loss (0 = off)

input group "=== Session rules ==="
input double       InpOrderExpiryHours = 4.0;    // Cancel unfilled orders this long after the bracket
input bool         InpFlatAnchorNY     = true;   // Anchor the flatten to the NY halt (DST-aware)
input int          InpFlatLeadMinutes  = 5;      // ...this many minutes before the 16:58 NY halt
input int          InpDailyFlatUTCH    = 21;     // Fixed-UTC fallback, only when InpFlatAnchorNY = false
input int          InpDailyFlatUTCM    = 50;     // Fixed-UTC fallback minute
input int          InpFridayCloseUTCH  = 20;     // Friday: flatten at this UTC hour (24 = off)
input double       InpMinRangePct      = 0.25;   // Skip brackets narrower than this % of price
input double       InpMaxRangePct      = 2.00;   // Skip brackets wider than this % of price
input bool         InpSkipSunday       = true;   // Skip the thin Sunday session
input bool         InpAdjustBidToMid   = true;   // Bars are bid-priced; shift to mid

input group "=== Bookkeeping ==="
input ulong        InpMagicBase        = 8830000; // Magic numbers base+0 .. base+7
input int          InpSlippagePoints   = 20;      // Max deviation on market close-outs
input bool         InpVerboseLog       = true;    // Narrate decisions to the Experts log

//+------------------------------------------------------------------+
//| Per-window definition and daily state                            |
//+------------------------------------------------------------------+
struct SessionWindow
{
   string   name;
   int      hour;          // UTC hour the bracket starts
   int      rangeMin;      // bracket length, minutes
   double   targetMult;    // take profit as a multiple of bracket width
   ulong    magic;

   int      armedDay;      // UTC day key the orders were placed
   int      doneDay;       // UTC day key this window already traded
   int      dataWarnDay;   // UTC day key we last complained about missing M1
   double   hi;            // bracket high (mid)
   double   lo;            // bracket low  (mid)
   datetime expiryUTC;
};

SessionWindow g_win[MAX_WINDOWS];
int           g_count = 0;
CTrade        g_trade;
CSymbolInfo   g_sym;

double g_volScale       = 1.0;
int    g_volScaleDay    = -1;
double g_dayStartEquity = 0.0;
int    g_equityDay      = -1;
bool   g_haltedToday    = false;

//+------------------------------------------------------------------+
//| Clock - the whole strategy is defined in UTC                     |
//+------------------------------------------------------------------+
datetime UTCNow()                        { return TimeCurrent() - (datetime)(InpGMTOffsetHours * 3600); }
datetime ServerFromUTC(const datetime u) { return u + (datetime)(InpGMTOffsetHours * 3600); }

//--- The Exness XAUUSD daily break is anchored to 17:00 New York, so in UTC
//--- it moves with US DST: 20:58->~22:01 in summer, 21:58->~23:01 in winter.
//--- Verified per-day against raw ticks over 2024-2026 (513 halt days): the
//--- last tick lands at HH:57:58 in 87% of sessions, never after HH:57:59.
//--- The server CLOCK is UTC year-round - it is the SESSION SCHEDULE that
//--- moves. Both are true; do not conflate them.
datetime NthSundayUTC(const int year, const int month, const int nth)
{
   MqlDateTime d;
   d.year = year; d.mon = month; d.day = 1;
   d.hour = 0;    d.min = 0;     d.sec = 0;
   MqlDateTime f;
   TimeToStruct(StructToTime(d), f);
   d.day = 1 + ((7 - f.day_of_week) % 7) + 7 * (nth - 1);
   return StructToTime(d);
}

//--- US DST: second Sunday in March to first Sunday in November.
bool IsUSDST(const datetime utc)
{
   MqlDateTime d;
   TimeToStruct(utc, d);
   return (utc >= NthSundayUTC(d.year, 3, 2) && utc < NthSundayUTC(d.year, 11, 1));
}

//--- UTC second-of-day at which the daily flatten fires.
int DailyFlatSecUTC(const datetime utc)
{
   if(!InpFlatAnchorNY)
      return InpDailyFlatUTCH * 3600 + InpDailyFlatUTCM * 60;
   const int halt = IsUSDST(utc) ? (20 * 3600 + 58 * 60) : (21 * 3600 + 58 * 60);
   return halt - InpFlatLeadMinutes * 60;
}

//--- YYYYMMDD in UTC; never repeats across years the way day_of_year does
int DayKeyOf(const datetime utc)
{
   MqlDateTime dt;
   TimeToStruct(utc, dt);
   return dt.year * 10000 + dt.mon * 100 + dt.day;
}
int UTCDayKey() { return DayKeyOf(UTCNow()); }

//--- Terminal-persistent names. The day's opening equity is the only piece of
//--- state no order carries, so it is the only thing that has to be stored.
//--- Keyed by magic base so two instances never share a baseline.
string DayKeyVarName()    { return StringFormat("SBDEP_%I64u_daykey",  InpMagicBase); }
string DayEquityVarName() { return StringFormat("SBDEP_%I64u_starteq", InpMagicBase); }

datetime UTCTodayAtHour(const int hour)
{
   MqlDateTime dt;
   TimeToStruct(UTCNow(), dt);
   dt.hour = hour; dt.min = 0; dt.sec = 0;
   return StructToTime(dt);
}

int UTCHourOf(const datetime utc)   { MqlDateTime d; TimeToStruct(utc, d); return d.hour; }
bool IsUTCSunday(const datetime utc){ MqlDateTime d; TimeToStruct(utc, d); return (d.day_of_week == 0); }
bool IsUTCFriday(const datetime utc){ MqlDateTime d; TimeToStruct(utc, d); return (d.day_of_week == 5); }

//+------------------------------------------------------------------+
//| Presets                                                          |
//+------------------------------------------------------------------+
void AddWindow(const string name, const int hour, const int rangeMin, const double tgt)
{
   if(g_count >= MAX_WINDOWS)
      return;
   const int i = g_count++;
   g_win[i].name        = name;
   g_win[i].hour        = hour;
   g_win[i].rangeMin    = rangeMin;
   g_win[i].targetMult  = (InpTargetMultOverride > 0.0) ? InpTargetMultOverride : tgt;
   g_win[i].magic       = InpMagicBase + (ulong)i;
   g_win[i].armedDay    = -1;
   g_win[i].doneDay     = -1;
   g_win[i].dataWarnDay = -1;
   g_win[i].hi          = 0.0;
   g_win[i].lo          = 0.0;
   g_win[i].expiryUTC   = 0;
}

string LoadPreset(const ENUM_PRESET p)
{
   g_count = 0;
   switch(p)
   {
      case PRESET_DEPLOY:
         //--- Same hours as GEO_2026_NO_H13. The hour set is NOT well supported
         //--- - dev and 2024-2026 rank the 21 tradable hours at only +0.26
         //--- correlation, and hours 1 and 5 were 12th and 20th on dev - but
         //--- re-picking them would be fitting the same noise a second time,
         //--- so they are left exactly as they were.
         AddWindow("h00_r60_t4",  0, 60, 4.0);
         AddWindow("h01_r60_t4",  1, 60, 4.0);
         AddWindow("h02_r60_t4",  2, 60, 4.0);
         AddWindow("h04_r60_t4",  4, 60, 4.0);
         AddWindow("h05_r60_t4",  5, 60, 4.0);
         AddWindow("h06_r60_t4",  6, 60, 4.0);
         AddWindow("h14_r60_t4", 14, 60, 4.0);
         return "DEPLOY (7 windows, 60m bracket, 4x target)";

      case PRESET_GEO_2026_NO_H13:
         AddWindow("h00_r60_t3",  0, 60, 3.0);
         AddWindow("h01_r60_t3",  1, 60, 3.0);
         AddWindow("h02_r60_t3",  2, 60, 3.0);
         AddWindow("h04_r60_t3",  4, 60, 3.0);
         AddWindow("h05_r60_t3",  5, 60, 3.0);
         AddWindow("h06_r60_t3",  6, 60, 3.0);
         AddWindow("h14_r60_t3", 14, 60, 3.0);
         return "GEO_2026_NO_H13 (7 windows)";

      case PRESET_GEO_2026:
         AddWindow("h00_r60_t3",  0, 60, 3.0);
         AddWindow("h01_r60_t3",  1, 60, 3.0);
         AddWindow("h02_r60_t3",  2, 60, 3.0);
         AddWindow("h04_r60_t3",  4, 60, 3.0);
         AddWindow("h05_r60_t3",  5, 60, 3.0);
         AddWindow("h06_r60_t3",  6, 60, 3.0);
         AddWindow("h13_r60_t3", 13, 60, 3.0);
         AddWindow("h14_r60_t3", 14, 60, 3.0);
         return "GEO_2026 (8 windows, as tested)";

      case PRESET_TOP8_2026:
         AddWindow("h00_r30_t3",  0, 30, 3.0);
         AddWindow("h01_r60_t3",  1, 60, 3.0);
         AddWindow("h02_r30_t3",  2, 30, 3.0);
         AddWindow("h05_r30_t3",  5, 30, 3.0);
         AddWindow("h08_r60_t3",  8, 60, 3.0);
         AddWindow("h14_r60_t3", 14, 60, 3.0);
         AddWindow("h15_r60_t3", 15, 60, 3.0);
         AddWindow("h18_r60_t3", 18, 60, 3.0);
         return "TOP8_2026 (8 windows, 2026 hour re-selection)";

      default:
         AddWindow("h00_r30_t1",  0, 30, 1.0);
         AddWindow("h01_r60_t3",  1, 60, 3.0);
         AddWindow("h02_r15_t3",  2, 15, 3.0);
         AddWindow("h04_r30_t3",  4, 30, 3.0);
         AddWindow("h05_r60_t2",  5, 60, 2.0);
         AddWindow("h06_r60_t3",  6, 60, 3.0);
         AddWindow("h13_r30_t3", 13, 30, 3.0);
         AddWindow("h14_r15_t2", 14, 15, 2.0);
         return "ORIGINAL (2024-2025 configuration)";
   }
}

//+------------------------------------------------------------------+
int OnInit()
{
   if(!g_sym.Name(_Symbol))
   {
      Print("ERROR: cannot select symbol ", _Symbol);
      return INIT_FAILED;
   }
   g_sym.RefreshRates();

   g_trade.SetDeviationInPoints(InpSlippagePoints);
   g_trade.SetTypeFillingBySymbol(_Symbol);
   g_trade.SetAsyncMode(false);

   const string preset = LoadPreset(InpPreset);

   PrintFormat("Session breakout DEPLOY on %s | preset %s | server %s | assumed GMT%+d | UTC %s",
               _Symbol, preset,
               TimeToString(TimeCurrent(), TIME_DATE | TIME_SECONDS),
               InpGMTOffsetHours,
               TimeToString(UTCNow(), TIME_DATE | TIME_SECONDS));

   if(InpAutoDetectGMT && !MQLInfoInteger(MQL_TESTER))
   {
      const int detected = (int)MathRound((double)(TimeCurrent() - TimeGMT()) / 3600.0);
      if(detected != InpGMTOffsetHours)
         PrintFormat("WARNING: terminal suggests the server is GMT%+d but InpGMTOffsetHours=%d. "
                     "Verify before trading.", detected, InpGMTOffsetHours);
   }

   //--- Sizing. Announce it in money terms, because "0.25%" and "what that
   //--- actually costs on a typical bracket" are very different sentences.
   if(InpUseRiskSizing)
   {
      const double eq   = AccountInfoDouble(ACCOUNT_EQUITY);
      const double risk = eq * InpRiskPctPerTrade / 100.0;
      PrintFormat("Sizing from the stop: %.2f%% of %.2f equity = %.2f risked per trade.",
                  InpRiskPctPerTrade, eq, risk);
      const double px = g_sym.Ask();
      if(px > 0.0)
      {
         const double typicalWidth = px * 0.004;   // a 0.4% bracket, the 2026 median
         const double lots = LotsForRisk(typicalWidth);
         PrintFormat("  a typical %.2f-wide bracket would size to %.2f lots", typicalWidth, lots);
         if(lots <= 0.0)
            PrintFormat("WARNING: that is below the broker minimum of %.2f lots, so trades will "
                        "be SKIPPED rather than over-risked. Raise InpRiskPctPerTrade or fund "
                        "the account further.", g_sym.LotsMin());
      }
      if(InpUseVolTargeting)
         Print("WARNING: InpUseVolTargeting is on together with risk sizing. That double-counts "
               "volatility - the bracket width already carries it. Turn one of them off.");
   }
   else
   {
      Print("WARNING: risk sizing is OFF, so every trade uses a flat ", InpBaseLots,
            " lots. The stop is the bracket width, which varies about 14x, so risk per "
            "trade will vary by the same factor. This is the 2026 build's behaviour.");
      if(InpBaseLots < g_sym.LotsMin())
         PrintFormat("WARNING: InpBaseLots %.4f is under the broker minimum %.4f. Unlike the "
                     "2026 build, which raised it, this build SKIPS the trade rather than "
                     "take more risk than asked for.", InpBaseLots, g_sym.LotsMin());
   }

   if(InpMaxDailyLossPct <= 0.0)
      Print("WARNING: the daily loss halt is disabled (InpMaxDailyLossPct = 0).");
   if(InpMaxOpenPositions >= g_count)
      PrintFormat("NOTE: InpMaxOpenPositions (%d) is at or above the window count (%d), so it "
                  "never binds - up to %d correlated %s positions can be open at once. That is "
                  "what was backtested. Lowering it to 4-5 is the risk-controlled choice but "
                  "was NOT tested, and it introduces a first-come-first-served bias between "
                  "windows.", InpMaxOpenPositions, g_count, g_count, _Symbol);

   if(InpMinRangePct < 0.20)
      PrintFormat("WARNING: InpMinRangePct is %.2f%%. Below about 0.20%% the fixed round-turn "
                  "cost eats the bracket: on the narrowest quintile it is 0.117R against "
                  "0.027R on the widest, which is what made the 2020-2023 period a "
                  "significant loser.", InpMinRangePct);

   //--- a flatten time earlier than the last bracket would silently kill windows
   int lastArm = 0;
   for(int i = 0; i < g_count; i++)
      lastArm = MathMax(lastArm, g_win[i].hour * 60 + g_win[i].rangeMin);
   const int flatMin = DailyFlatSecUTC(UTCNow()) / 60;
   const int haltMin = flatMin + (InpFlatAnchorNY ? InpFlatLeadMinutes : 0);
   if(flatMin <= lastArm)
      PrintFormat("WARNING: daily flatten %02d:%02d is at or before the last bracket closes "
                  "(%02d:%02d) - those windows will never trade.",
                  flatMin / 60, flatMin % 60, lastArm / 60, lastArm % 60);
   else if(InpFlatAnchorNY)
      PrintFormat("Daily flatten %02d:%02d UTC = %d min before the %02d:%02d halt "
                  "(16:58 New York; US DST %s today; last bracket closes %02d:%02d)",
                  flatMin / 60, flatMin % 60, InpFlatLeadMinutes,
                  haltMin / 60, haltMin % 60, IsUSDST(UTCNow()) ? "ON" : "off",
                  lastArm / 60, lastArm % 60);
   else
      PrintFormat("WARNING: fixed-UTC flatten %02d:%02d - this lands inside the summer halt "
                  "(20:58-22:01) and will pay swap on ~66%% of days. Last bracket %02d:%02d.",
                  flatMin / 60, flatMin % 60, lastArm / 60, lastArm % 60);

   //--- Re-init is not rare and is mostly not chosen: a restart, a reconnect,
   //--- a recompile and an edit to the inputs all land here. Rebuild the
   //--- day's state from the market BEFORE the first tick can re-arm anything.
   RollDailyState(UTCDayKey());
   AdoptExistingState();
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason) { }

//+------------------------------------------------------------------+
void OnTick()
{
   if(!g_sym.RefreshRates())
      return;

   const datetime utc   = UTCNow();
   const int      today = UTCDayKey();

   RollDailyState(today);
   EnforceDailyLossHalt();

   //--- Flatten rules, most reliable first. A clock-window check alone is
   //--- not enough: on Friday the market can close before the window is
   //--- reached, no tick lands inside it, and the position rides the gap.
   //--- The stale sweep is the backstop that makes daily-flat actually hold.
   if(CloseStaleFromEarlierSessions() > 0)
      return;

   if(InpFridayCloseUTCH < 24 && IsUTCFriday(utc) && UTCHourOf(utc) >= InpFridayCloseUTCH)
   {
      CloseEverything("friday cutoff");
      CancelAllPending("friday cutoff");
      return;
   }

   if(PastDailyFlatten(utc))
   {
      CloseEverything("daily flatten");      // also cancels every pending order
      return;
   }

   if(g_haltedToday)
   {
      CancelAllPending("daily loss halt");
      return;
   }

   for(int i = 0; i < g_count; i++)
      ProcessWindow(i, utc, today);
}

//+------------------------------------------------------------------+
void RollDailyState(const int today)
{
   if(g_equityDay == today)
      return;
   g_equityDay   = today;
   g_haltedToday = false;

   //--- Re-sampling opening equity on every re-init walks the daily-loss
   //--- baseline down with each restart: the halt would then measure from the
   //--- restart rather than from the day, which is silently no protection at
   //--- all. The halt FLAG needs no persistence - with the baseline correct,
   //--- EnforceDailyLossHalt re-trips on the next tick by itself.
   const string kDay = DayKeyVarName(), kEq = DayEquityVarName();
   if(GlobalVariableCheck(kDay) && (int)GlobalVariableGet(kDay) == today
      && GlobalVariableCheck(kEq) && GlobalVariableGet(kEq) > 0.0)
   {
      g_dayStartEquity = GlobalVariableGet(kEq);
      PrintFormat("--- resumed UTC day (%d), start equity %.2f restored ---",
                  today, g_dayStartEquity);
   }
   else
   {
      g_dayStartEquity = AccountInfoDouble(ACCOUNT_EQUITY);
      GlobalVariableSet(kDay, (double)today);
      GlobalVariableSet(kEq, g_dayStartEquity);
      if(InpVerboseLog)
         PrintFormat("--- new UTC day (%d), start equity %.2f ---", today, g_dayStartEquity);
   }

   for(int i = 0; i < g_count; i++)
      g_win[i].armedDay = -1;
}

void EnforceDailyLossHalt()
{
   if(InpMaxDailyLossPct <= 0.0 || g_dayStartEquity <= 0.0 || g_haltedToday)
      return;
   const double loss = (g_dayStartEquity - AccountInfoDouble(ACCOUNT_EQUITY))
                       / g_dayStartEquity * 100.0;
   if(loss >= InpMaxDailyLossPct)
   {
      g_haltedToday = true;
      PrintFormat("Daily loss %.2f%% hit the %.2f%% limit - halting for today.",
                  loss, InpMaxDailyLossPct);
      CloseEverything("daily loss halt");
   }
}

//--- true from the daily flatten time until the UTC day rolls over.
//---
//--- The flatten is anchored to the daily halt, not to a fixed UTC clock. A
//--- fixed 21:50 UTC lands INSIDE the summer halt (20:58-22:01) on ~66% of
//--- trading days. OnTick is tick-driven, so nothing fires during the halt
//--- and CloseEverything() slips to the first tick after the reopen - past
//--- the swap charge. Anchoring holds swap at exactly 0.
bool PastDailyFlatten(const datetime utc)
{
   MqlDateTime dt;
   TimeToStruct(utc, dt);
   const int sec = dt.hour * 3600 + dt.min * 60 + dt.sec;
   return (sec >= DailyFlatSecUTC(utc));
}

//+------------------------------------------------------------------+
void ProcessWindow(const int i, const datetime utc, const int today)
{
   ResolveOCO(i);
   if(g_win[i].armedDay == today && utc >= g_win[i].expiryUTC)
      CancelWindowPending(i, "expired unfilled");

   if(g_win[i].doneDay == today || g_win[i].armedDay == today)
      return;

   const datetime rangeStart = UTCTodayAtHour(g_win[i].hour);
   const datetime rangeEnd   = rangeStart + g_win[i].rangeMin * 60;
   if(utc < rangeEnd)
      return;

   if(utc >= rangeEnd + (datetime)(InpOrderExpiryHours * 3600))
   {
      g_win[i].armedDay = today;          // too late to arm; stop rechecking
      return;
   }
   if(InpSkipSunday && IsUTCSunday(utc))
   {
      g_win[i].armedDay = today;
      return;
   }

   double hi = 0.0, lo = 0.0;
   if(!BuildRange(rangeStart, rangeEnd, hi, lo))
   {
      //--- Do NOT mark the window armed here. CopyRates returns nothing while
      //--- the terminal is still backfilling M1 history, which is normal after
      //--- a restart; the 2026 build treated that as "no data today" and
      //--- silently sat out the session. Retry on the next tick instead - the
      //--- expiry check above is what eventually gives up. Complain once a day.
      if(InpVerboseLog && g_win[i].dataWarnDay != today)
      {
         g_win[i].dataWarnDay = today;
         PrintFormat("%s: no M1 data for the bracket yet - will retry until %s",
                     g_win[i].name,
                     TimeToString(rangeEnd + (datetime)(InpOrderExpiryHours * 3600),
                                  TIME_MINUTES));
      }
      return;
   }

   const double width    = hi - lo;
   const double refPx    = (hi + lo) * 0.5;
   const double widthPct = (refPx > 0.0) ? width / refPx * 100.0 : 0.0;

   if(widthPct < InpMinRangePct || widthPct > InpMaxRangePct)
   {
      if(InpVerboseLog)
         PrintFormat("%s: bracket %.2f (%.3f%%) outside [%.2f%%, %.2f%%] - skipping",
                     g_win[i].name, width, widthPct, InpMinRangePct, InpMaxRangePct);
      g_win[i].armedDay = today;
      return;
   }
   if(!SpreadAcceptable())
   {
      if(InpVerboseLog)
         PrintFormat("%s: spread %.3f over the %.3f limit - skipping",
                     g_win[i].name, g_sym.Ask() - g_sym.Bid(), InpMaxSpreadUSD);
      g_win[i].armedDay = today;
      return;
   }
   //--- on a restart price may already have broken out; arming only the far
   //--- side would take the wrong direction, so sit the day out
   if(g_sym.Ask() >= hi || g_sym.Bid() <= lo)
   {
      if(InpVerboseLog)
         PrintFormat("%s: price already outside %.2f-%.2f - missed the break, skipping",
                     g_win[i].name, lo, hi);
      g_win[i].armedDay = today;
      return;
   }
   if(CountOurPositions() >= InpMaxOpenPositions)
   {
      g_win[i].armedDay = today;
      return;
   }

   PlaceBracket(i, hi, lo, width, rangeEnd, today);
}

//+------------------------------------------------------------------+
//| Bracket = high/low of M1 bars over [start, end) in UTC.          |
//| MT5 bars are bid-priced while the study measured mid, so both     |
//| extremes shift up by half the spread. Width is unaffected.        |
//+------------------------------------------------------------------+
bool BuildRange(const datetime rangeStartUTC, const datetime rangeEndUTC,
                double &hi, double &lo)
{
   MqlRates rates[];
   ArraySetAsSeries(rates, false);
   const datetime from = ServerFromUTC(rangeStartUTC);
   const datetime to   = ServerFromUTC(rangeEndUTC) - 60;

   const int n = CopyRates(_Symbol, PERIOD_M1, from, to, rates);
   if(n <= 0)
      return false;

   hi = rates[0].high;
   lo = rates[0].low;
   for(int k = 1; k < n; k++)
   {
      if(rates[k].high > hi) hi = rates[k].high;
      if(rates[k].low  < lo) lo = rates[k].low;
   }
   if(InpAdjustBidToMid)
   {
      const double half = (g_sym.Ask() - g_sym.Bid()) * 0.5;
      hi += half;
      lo += half;
   }
   hi = g_sym.NormalizePrice(hi);
   lo = g_sym.NormalizePrice(lo);
   return (hi > lo);
}

//+------------------------------------------------------------------+
//| A Buy Stop triggers on ask and a Sell Stop on bid, which is       |
//| exactly how the backtest detected breaks - no adjustment needed.  |
//+------------------------------------------------------------------+
void PlaceBracket(const int i, const double hi, const double lo, const double width,
                  const datetime rangeEnd, const int today)
{
   //--- The stop is the opposite edge, so the width IS the risk. Size from it.
   const double lots = ResolveLots(width);
   if(lots <= 0.0)
   {
      if(InpVerboseLog)
      {
         if(InpUseRiskSizing)
            PrintFormat("%s: a %.2f-wide bracket at %.2f%% risk sizes below the broker minimum "
                        "of %.2f lots - skipping rather than over-risking",
                        g_win[i].name, width, InpRiskPctPerTrade, g_sym.LotsMin());
         else
            PrintFormat("%s: %.4f lots is below the broker minimum of %.2f - skipping",
                        g_win[i].name, InpBaseLots, g_sym.LotsMin());
      }
      g_win[i].armedDay = today;
      return;
   }

   const double stopsLevel = (double)SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL)
                             * g_sym.Point();
   const bool buyOK  = (hi - g_sym.Ask() >= stopsLevel);
   const bool sellOK = (g_sym.Bid() - lo >= stopsLevel);
   if(!buyOK && !sellOK)
   {
      if(InpVerboseLog)
         PrintFormat("%s: both sides inside the %.2f stops level - skipping",
                     g_win[i].name, stopsLevel);
      g_win[i].armedDay = today;
      return;
   }

   const double tpBuy  = g_sym.NormalizePrice(hi + g_win[i].targetMult * width);
   const double tpSell = g_sym.NormalizePrice(lo - g_win[i].targetMult * width);

   g_trade.SetExpertMagicNumber(g_win[i].magic);
   const string tag = g_win[i].name;
   bool placed = false;

   if(buyOK)
   {
      if(g_trade.BuyStop(lots, hi, _Symbol, lo, tpBuy, ORDER_TIME_GTC, 0, tag))
         placed = true;
      else
         PrintFormat("%s: BuyStop failed (%d) %s", tag,
                     g_trade.ResultRetcode(), g_trade.ResultRetcodeDescription());
   }
   if(sellOK)
   {
      if(g_trade.SellStop(lots, lo, _Symbol, hi, tpSell, ORDER_TIME_GTC, 0, tag))
         placed = true;
      else
         PrintFormat("%s: SellStop failed (%d) %s", tag,
                     g_trade.ResultRetcode(), g_trade.ResultRetcodeDescription());
   }

   g_win[i].hi        = hi;
   g_win[i].lo        = lo;
   g_win[i].expiryUTC = rangeEnd + (datetime)(InpOrderExpiryHours * 3600);
   g_win[i].armedDay  = today;

   if(placed && InpVerboseLog)
      PrintFormat("%s armed: %.2f-%.2f (width %.2f, %.3f%%), %.2f lots, targets %.2f / %.2f",
                  tag, lo, hi, width, width / ((hi + lo) * 0.5) * 100.0, lots, tpBuy, tpSell);
}

//+------------------------------------------------------------------+
//| Manual OCO: first fill retires the other side, then the take      |
//| profit is re-anchored to the price actually obtained.             |
//|                                                                   |
//| The race here is real in principle - the fill has to be SEEN      |
//| before the resting order can be deleted - but it was measured     |
//| across 4,611 gold trades in 2024-2026 and the opposite bracket    |
//| edge was never touched within 30 seconds of a fill. The other     |
//| side is a full bracket width away.                                |
//+------------------------------------------------------------------+
void ResolveOCO(const int i)
{
   const ulong pos = FindPosition(g_win[i].magic);
   if(pos == 0)
      return;

   for(int k = OrdersTotal() - 1; k >= 0; k--)
   {
      const ulong t = OrderGetTicket(k);
      if(t == 0) continue;
      if(OrderGetString(ORDER_SYMBOL) != _Symbol) continue;   // was missing
      if((ulong)OrderGetInteger(ORDER_MAGIC) != g_win[i].magic) continue;
      g_trade.OrderDelete(t);
   }
   g_win[i].doneDay = UTCDayKey();
   SyncTakeProfit(i, pos);
}

void SyncTakeProfit(const int i, const ulong ticket)
{
   if(!PositionSelectByTicket(ticket))
      return;
   const double width = g_win[i].hi - g_win[i].lo;
   if(width <= 0.0)
      return;

   const double entry = PositionGetDouble(POSITION_PRICE_OPEN);
   const double sl    = PositionGetDouble(POSITION_SL);
   const double tpNow = PositionGetDouble(POSITION_TP);
   const long   type  = PositionGetInteger(POSITION_TYPE);

   const double tpWant = (type == POSITION_TYPE_BUY)
                         ? g_sym.NormalizePrice(entry + g_win[i].targetMult * width)
                         : g_sym.NormalizePrice(entry - g_win[i].targetMult * width);

   if(MathAbs(tpWant - tpNow) < g_sym.Point())
      return;

   g_trade.SetExpertMagicNumber(g_win[i].magic);
   if(g_trade.PositionModify(ticket, sl, tpWant) && InpVerboseLog)
      PrintFormat("%s: filled at %.2f, take profit re-anchored to %.2f",
                  g_win[i].name, entry, tpWant);
}

//+------------------------------------------------------------------+
//| Sizing                                                            |
//|                                                                   |
//| The stop is the opposite edge of the bracket, so the bracket      |
//| width is exactly what one unit of risk costs. Sizing from it is   |
//| the whole point of this build: measured over 2024-2026, a flat    |
//| lot size put between $4.78 and $70.31 at risk on the same         |
//| nominal position - a 14x spread between the 5th and 95th          |
//| percentile.                                                       |
//|                                                                   |
//| Loss per lot comes from the broker's own tick value rather than   |
//| an assumed 100 oz contract, so this is correct on any symbol and  |
//| does not inherit the "contract size is inferred, not published"   |
//| caveat that every pip-denominated figure in the study carries.    |
//+------------------------------------------------------------------+
double LotsForRisk(const double stopDistance)
{
   if(stopDistance <= 0.0)
      return 0.0;
   const double tickSize  = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   const double tickValue = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   if(tickSize <= 0.0 || tickValue <= 0.0)
   {
      Print("ERROR: broker reports no tick size/value for ", _Symbol,
            " - cannot size from the stop.");
      return 0.0;
   }
   const double lossPerLot = stopDistance / tickSize * tickValue;
   if(lossPerLot <= 0.0)
      return 0.0;
   const double riskUSD = AccountInfoDouble(ACCOUNT_EQUITY) * InpRiskPctPerTrade / 100.0;
   return NormalizeLots(riskUSD / lossPerLot);
}

double ResolveLots(const double stopDistance)
{
   if(InpUseRiskSizing)
      return LotsForRisk(stopDistance);

   //--- Legacy path: flat lots, optionally scaled by trailing volatility.
   double lots = InpBaseLots;
   if(InpUseVolTargeting)
   {
      const int today = UTCDayKey();
      if(g_volScaleDay != today)
      {
         const double tv = TrailingRealizedVol(InpVolLookbackDays);
         g_volScale = (tv > 0.0)
                      ? MathMax(1.0 / InpMaxVolScale, MathMin(InpMaxVolScale, InpTargetVolPct / tv))
                      : 1.0;
         g_volScaleDay = today;
         if(InpVerboseLog)
            PrintFormat("Volatility scale today: %.3f (trailing %.3f%% vs target %.3f%%)",
                        g_volScale, tv, InpTargetVolPct);
      }
      lots = InpBaseLots * g_volScale;
   }
   return NormalizeLots(lots);
}

double TrailingRealizedVol(const int days)
{
   double sum = 0.0;
   int    used = 0;
   for(int back = 1; back <= days * 2 + 10 && used < days; back++)
   {
      const double v = DailyRealizedVol(UTCTodayAtHour(0) - (datetime)(back * 86400));
      if(v > 0.0) { sum += v; used++; }
   }
   return (used >= 5) ? sum / used : 0.0;
}

double DailyRealizedVol(const datetime dayStartUTC)
{
   MqlRates rates[];
   ArraySetAsSeries(rates, false);
   const int n = CopyRates(_Symbol, PERIOD_M5, ServerFromUTC(dayStartUTC),
                           ServerFromUTC(dayStartUTC + 86400) - 300, rates);
   if(n < 30)
      return 0.0;

   double r[];
   ArrayResize(r, n - 1);
   double mean = 0.0;
   int    m    = 0;
   for(int k = 1; k < n; k++)
   {
      if(rates[k - 1].close <= 0.0 || rates[k].close <= 0.0) continue;
      r[m] = MathLog(rates[k].close / rates[k - 1].close);
      mean += r[m];
      m++;
   }
   if(m < 20)
      return 0.0;
   mean /= m;

   double ss = 0.0;
   for(int k = 0; k < m; k++)
      ss += (r[k] - mean) * (r[k] - mean);
   return MathSqrt(ss / (m - 1)) * MathSqrt((double)m) * 100.0;
}

//--- Floors to the lot step rather than rounding, so rounding can never push
//--- the position above the intended risk. Returns 0 - meaning "skip the
//--- trade" - when the honest size is below the broker minimum, because
//--- raising it to the minimum would silently take more risk than asked for.
double NormalizeLots(double lots)
{
   const double step = g_sym.LotsStep();
   int digits = 2;
   if(step > 0.0)
   {
      //--- the epsilon matters: 0.02/0.01 evaluates to 1.9999... in binary
      //--- floating point, and a bare MathFloor would silently drop a step
      lots = MathFloor(lots / step + 1e-9) * step;
      digits = (int)MathMax(0.0, MathCeil(-MathLog10(step)));
   }
   if(lots < g_sym.LotsMin() - 1e-9)
      return 0.0;
   return NormalizeDouble(MathMin(lots, g_sym.LotsMax()), digits);
}

//+------------------------------------------------------------------+
//| Housekeeping                                                     |
//+------------------------------------------------------------------+
bool IsOurMagic(const ulong m) { return (m >= InpMagicBase && m < InpMagicBase + MAX_WINDOWS); }

bool SpreadAcceptable()
{
   if(InpMaxSpreadUSD <= 0.0)
      return true;
   return ((g_sym.Ask() - g_sym.Bid()) <= InpMaxSpreadUSD);
}

ulong FindPosition(const ulong magic)
{
   for(int k = PositionsTotal() - 1; k >= 0; k--)
   {
      const ulong t = PositionGetTicket(k);
      if(t == 0) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
      if((ulong)PositionGetInteger(POSITION_MAGIC) == magic)
         return t;
   }
   return 0;
}

int CountOurPositions()
{
   int n = 0;
   for(int k = PositionsTotal() - 1; k >= 0; k--)
   {
      if(PositionGetTicket(k) == 0) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
      if(IsOurMagic((ulong)PositionGetInteger(POSITION_MAGIC)))
         n++;
   }
   return n;
}

//+------------------------------------------------------------------+
//| Rebuild per-window state from what is already in the market.      |
//|                                                                   |
//| MT5 destroys and recreates the EA on a timeframe change, a symbol |
//| change, a recompile, an edit to the inputs, and every terminal    |
//| restart. Globals go back to their initialisers - the orders and   |
//| positions do not, because they live on the server. Without this   |
//| pass the EA sees an unarmed window, re-arms it, and stacks a      |
//| second bracket at the SAME two prices. The stacked pairs then     |
//| fill together on the break: N re-inits, N times the intended      |
//| size, against a fixed daily loss limit. That is an account        |
//| killer, and it fires on restarts nobody chose.                    |
//|                                                                   |
//| Nothing has to be persisted to undo it. Everything lost is        |
//| recoverable from the market itself:                               |
//|   hi / lo    ARE the pending prices - a BuyStop sits at hi        |
//|   expiryUTC  recomputes from the window's own hour and length     |
//|   armedDay   a live pending means this window armed today         |
//|   doneDay    an open position means it has already traded         |
//+------------------------------------------------------------------+
void AdoptExistingState()
{
   const int today = UTCDayKey();
   int adopted = 0, stale = 0, dupes = 0;

   for(int i = 0; i < g_count; i++)
   {
      double hi = 0.0, lo = 0.0;
      ulong  buyTicket = 0, sellTicket = 0;
      int    pend = 0;

      for(int k = OrdersTotal() - 1; k >= 0; k--)
      {
         const ulong t = OrderGetTicket(k);
         if(t == 0) continue;
         if(OrderGetString(ORDER_SYMBOL) != _Symbol) continue;
         if((ulong)OrderGetInteger(ORDER_MAGIC) != g_win[i].magic) continue;

         //--- an order that outlived a terminal outage: the daily flatten
         //--- never got a tick to cancel it
         const datetime setUTC = (datetime)OrderGetInteger(ORDER_TIME_SETUP)
                                 - (datetime)(InpGMTOffsetHours * 3600);
         if(DayKeyOf(setUTC) != today)
         {
            if(g_trade.OrderDelete(t))
               stale++;
            continue;
         }

         const long ty = OrderGetInteger(ORDER_TYPE);
         if(ty != ORDER_TYPE_BUY_STOP && ty != ORDER_TYPE_SELL_STOP)
            continue;

         //--- Duplicates can only exist because an earlier build re-armed over
         //--- a live bracket. Keep one of each side and delete the rest.
         const bool isBuy = (ty == ORDER_TYPE_BUY_STOP);
         if((isBuy && buyTicket != 0) || (!isBuy && sellTicket != 0))
         {
            if(g_trade.OrderDelete(t))
               dupes++;
            continue;
         }
         if(isBuy)
         {
            buyTicket = t;
            hi = OrderGetDouble(ORDER_PRICE_OPEN);
         }
         else
         {
            sellTicket = t;
            lo = OrderGetDouble(ORDER_PRICE_OPEN);
         }
         pend++;
      }

      const ulong pos = FindPosition(g_win[i].magic);
      if(pend == 0 && pos == 0)
         continue;

      g_win[i].armedDay  = today;
      g_win[i].expiryUTC = UTCTodayAtHour(g_win[i].hour)
                           + (datetime)(g_win[i].rangeMin * 60)
                           + (datetime)(InpOrderExpiryHours * 3600);
      if(pos != 0)
         g_win[i].doneDay = today;
      if(hi > 0.0 && lo > 0.0 && hi > lo)
      {
         g_win[i].hi = hi;
         g_win[i].lo = lo;
      }
      adopted++;

      PrintFormat("%s: adopted %d pending%s%s", g_win[i].name, pend,
                  (pos != 0) ? " and an open position" : "",
                  (g_win[i].hi > 0.0)
                  ? StringFormat(", bracket %.2f-%.2f restored", g_win[i].lo, g_win[i].hi)
                  : ", bracket not reconstructible - take profit left as placed");
   }

   if(adopted > 0 || stale > 0 || dupes > 0)
      PrintFormat("Re-init: %d window(s) adopted, %d stale order(s) and %d duplicate(s) "
                  "cancelled. Without this pass a second bracket would have been "
                  "stacked on each adopted window.", adopted, stale, dupes);
}

//--- Close anything opened on an earlier UTC day. This is the backstop that
//--- makes the daily-flat rule hold across weekends and holidays, when no
//--- tick arrives before midnight.
int CloseStaleFromEarlierSessions()
{
   const int today = UTCDayKey();
   int closed = 0;
   for(int k = PositionsTotal() - 1; k >= 0; k--)
   {
      const ulong t = PositionGetTicket(k);
      if(t == 0) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
      const ulong magic = (ulong)PositionGetInteger(POSITION_MAGIC);
      if(!IsOurMagic(magic)) continue;

      const datetime openedUTC = (datetime)PositionGetInteger(POSITION_TIME)
                                 - (datetime)(InpGMTOffsetHours * 3600);
      if(DayKeyOf(openedUTC) == today)
         continue;

      g_trade.SetExpertMagicNumber(magic);
      if(g_trade.PositionClose(t))
      {
         closed++;
         PrintFormat("Stale position %I64u from %s closed - it survived a session boundary",
                     t, TimeToString(openedUTC, TIME_DATE | TIME_MINUTES));
      }
   }
   return closed;
}

void CancelWindowPending(const int i, const string why)
{
   for(int k = OrdersTotal() - 1; k >= 0; k--)
   {
      const ulong t = OrderGetTicket(k);
      if(t == 0) continue;
      if(OrderGetString(ORDER_SYMBOL) != _Symbol) continue;   // was missing
      if((ulong)OrderGetInteger(ORDER_MAGIC) != g_win[i].magic) continue;
      if(g_trade.OrderDelete(t) && InpVerboseLog)
         PrintFormat("%s: pending order cancelled (%s)", g_win[i].name, why);
   }
   g_win[i].doneDay = UTCDayKey();
}

void CancelAllPending(const string why)
{
   for(int k = OrdersTotal() - 1; k >= 0; k--)
   {
      const ulong t = OrderGetTicket(k);
      if(t == 0) continue;
      if(OrderGetString(ORDER_SYMBOL) != _Symbol) continue;
      if(!IsOurMagic((ulong)OrderGetInteger(ORDER_MAGIC))) continue;
      g_trade.OrderDelete(t);
   }
}

void CloseEverything(const string why)
{
   CancelAllPending(why);
   for(int k = PositionsTotal() - 1; k >= 0; k--)
   {
      const ulong t = PositionGetTicket(k);
      if(t == 0) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
      const ulong magic = (ulong)PositionGetInteger(POSITION_MAGIC);
      if(!IsOurMagic(magic)) continue;
      g_trade.SetExpertMagicNumber(magic);
      if(g_trade.PositionClose(t) && InpVerboseLog)
         PrintFormat("Position %I64u closed (%s)", t, why);
   }
}
//+------------------------------------------------------------------+

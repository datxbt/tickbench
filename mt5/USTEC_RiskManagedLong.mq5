//+------------------------------------------------------------------+
//|                                   USTEC_RiskManagedLong.mq5      |
//|                                                                  |
//|  Volatility-targeted, trend-gated long exposure to USTEC.        |
//|                                                                  |
//|  ================================================================|
//|  READ THIS BEFORE YOU TRADE IT                                   |
//|  ================================================================|
//|  THIS EXPERT HAS NO DIRECTIONAL EDGE AND DOES NOT CLAIM ONE.     |
//|                                                                  |
//|  Sixteen directional hypotheses were tested on 696M ticks of     |
//|  Exness USTEC data (2020-2026) before this was written. Every    |
//|  one of them failed: intraday seasonality (dev-selected windows  |
//|  scored Sharpe +0.81 in sample and -1.79 out of sample), the     |
//|  broker's session-halt gap (real until June 2023, then killed    |
//|  by a schedule change), minute-level autoregression (real at     |
//|  t = -29 but worth 0.05 bps against a 1.4 bps round turn),       |
//|  cross-asset lead-lag (every sign flipped between periods),      |
//|  gap continuation, turn-of-month, day-of-week and dip-buying.    |
//|                                                                  |
//|  What this expert trades instead is the equity risk premium,     |
//|  sized by a volatility forecast. On non-overlapping 20-session   |
//|  blocks, one block's realised volatility predicts the next -     |
//|  corr +0.45 dev, +0.23 validation, +0.15 test - while one        |
//|  block's return predicts the next at -0.05, -0.15, +0.08, which  |
//|  is nothing. Risk is forecastable; direction is not. That is     |
//|  enough to hold the same exposure at roughly constant risk       |
//|  instead of at whatever risk the market happens to be running.   |
//|                                                                  |
//|  Note the direction of travel: the volatility signal is weaker   |
//|  in the two recent periods than in dev. What stayed constant is  |
//|  the drawdown reduction, which is the thing being sold here.     |
//|                                                                  |
//|  So: if you would not hold a long Nasdaq position at all, this   |
//|  expert is not for you. What it offers someone who would is a    |
//|  materially smaller drawdown for the same return.                |
//|                                                                  |
//|  ================================================================|
//|  MEASURED RESULTS, ZERO SWAP, COSTS FROM THE MEASURED TAPE       |
//|  ================================================================|
//|                     this expert          buy and hold            |
//|                   CAGR   SR    maxDD    CAGR   SR    maxDD       |
//|   dev   2020-23  11.1%  0.95  -15.1%   13.2%  0.60  -39.8%       |
//|   val   2024-25  13.8%  0.91  -12.9%   20.0%  0.90  -24.9%       |
//|   test  25H2-26  18.9%  1.15   -7.9%   22.8%  1.20  -10.8%       |
//|                                                                  |
//|  The test row is a single run on a period held out and locked    |
//|  before any of this work started. It is now spent.               |
//|                                                                  |
//|  Read that table as: same risk-adjusted return, roughly half     |
//|  the drawdown, in all three periods. It does NOT beat buy and    |
//|  hold on raw return, and in a strong uninterrupted bull market   |
//|  it will lag - it lagged in 2020 (+6.7% against +34.3%) and in   |
//|  2026 (+6.5% against +12.6%). It earns its keep in 2022, where   |
//|  it lost 10.5% against buy and hold's 36.1%.                     |
//|                                                                  |
//|  ================================================================|
//|  THE ONE COST THAT IS NOT IN THAT TABLE: SWAP                    |
//|  ================================================================|
//|  A tick feed carries quotes, not financing, so overnight swap    |
//|  could not be measured and is NOT in the numbers above. It is    |
//|  the largest single uncertainty in this strategy.                |
//|                                                                  |
//|  Break-even swap - the rate at which the whole return goes to    |
//|  the broker - is 6.1 bps/night on dev, 4.8 on validation, 5.4    |
//|  on test. A typical index-CFD financing charge is 1.5-2.0        |
//|  bps/night, which takes roughly 5-7 percentage points a year     |
//|  off the CAGR. BEFORE DEPLOYING, read your account's actual      |
//|  swap for this symbol and put it in InpSwapCheckBps below; the   |
//|  expert refuses to trade if the live figure exceeds it.          |
//|                                                                  |
//|  Buy and hold pays the same swap on a larger average position    |
//|  (1.00 against 0.50-0.92 here), so the comparison above is not   |
//|  changed in buy and hold's favour by including it.               |
//|                                                                  |
//|  ================================================================|
//|  WHAT IT DOES, MECHANICALLY                                      |
//|  ================================================================|
//|  Once per day, at the US cash open (09:30 New York):             |
//|                                                                  |
//|    1. Build the series of US cash-session closes - the close of  |
//|       the M30 bar ending 16:00 New York, one per session.        |
//|    2. gate = (last cash close > SMA(cash closes, 200))           |
//|    3. vol  = stdev of the last 20 cash close-to-close log        |
//|              returns, annualised by sqrt(252)                    |
//|    4. w    = gate * clamp(TargetVol / vol, 0, MaxWeight)         |
//|    5. If |w - w_held| > Band, resize the position to w.          |
//|                                                                  |
//|  w is notional as a multiple of equity, so lots =                |
//|  w * equity / (price * contract size). Everything the signal     |
//|  uses closed before the decision; nothing reads the current bar. |
//|                                                                  |
//|  Turnover is about 5-7x notional a year - a handful of orders    |
//|  a month. That is why cost barely matters: tripling the spread   |
//|  and making every fill fully adverse moves the CAGR by 0.04      |
//|  percentage points. Do not "improve" this by rebalancing more    |
//|  often; the band is doing real work.                             |
//|                                                                  |
//|  ================================================================|
//|  HOW IT WILL FAIL                                                |
//|  ================================================================|
//|  * A gap through the trend gate. The gate reads a daily close;   |
//|    a crash that happens overnight is taken at full size.         |
//|  * Whipsaw around the 200-session average in a flat market:      |
//|    repeated in-out with no trend to pay for it.                  |
//|  * Leverage. In a quiet market the volatility target asks for    |
//|    more than 1x notional (it averaged 1.17x in 2025). MaxWeight  |
//|    caps this. Setting it above 2.0 is a decision about margin,   |
//|    not about the strategy.                                       |
//|  * Swap repricing. Financing is not fixed; a rate rise is a      |
//|    direct subtraction from the return.                           |
//|                                                                  |
//|  Research: reports/strategies/USTEC_RISK_MANAGED_LONG.md         |
//|  Model:    src/qlab/strategies/risk_managed_long.py              |
//+------------------------------------------------------------------+
#property copyright "QuantProject"
#property version   "1.00"
#property strict
#property description "Volatility-targeted, trend-gated long USTEC. No directional edge claimed."

#include <Trade\Trade.mqh>

//--- Signal -------------------------------------------------------------
input int    InpMaLen          = 200;    // Trend filter, in cash sessions
input int    InpVolLen         = 20;     // Volatility window, in cash sessions
input double InpTargetVolPct   = 15.0;   // Annualised volatility target, %
input double InpMaxWeight      = 2.0;    // Cap on notional / equity
input double InpGateFloor      = 0.0;    // Weight multiplier below the average
input double InpBand           = 0.10;   // No-trade band on the weight

//--- Session ------------------------------------------------------------
input int    InpCashOpenMin    = 570;    // US cash open, NY minutes (09:30)
input int    InpCashCloseMin   = 960;    // US cash close, NY minutes (16:00)
input int    InpDecisionWindow = 30;     // Minutes after the open to still act

//--- Risk and safety ----------------------------------------------------
input double InpMaxSpreadPts   = 5.0;    // Skip the rebalance above this spread
input double InpSwapCheckBps   = 2.5;    // Refuse to trade above this swap
input double InpKillDrawdownPct= 25.0;   // Flatten and stop below this equity DD
input double InpEquityOverride = 0.0;    // Size off this equity instead, 0 = live

//--- Prop-firm mode -----------------------------------------------------
//  A funded-account challenge is a barrier problem, not a Sharpe problem:
//  reach +X% before touching a daily floor or a static total floor. Two
//  things follow and both are switched on here.
//
//  1. InpPropScale trades a fraction of the strategy. Smaller size raises
//     the pass probability and costs only time - and FTMO has no deadline.
//  2. Cushion sizing. The total-loss floor is measured against the INITIAL
//     balance and never moves, so an account at +6% is 16% from the floor,
//     not 10%. Sizing on the cushion that is left means starting small and
//     growing only with room that has been earned. In continuous time it
//     makes the floor unreachable; gaps break that, so it is a large
//     reduction in ruin risk rather than a guarantee.
//
//  Measured on 2020-2026 USTEC with the strategy capped at 1.0x weight,
//  cushion sizing at 0.60x: P(pass both steps) 98-100%, median 1.8-3.0
//  years. At 1.00x it is 66-84% and 0.7-1.8 years. Pick your point on that
//  frontier deliberately - see reports/strategies/PROPFIRM_SIZING.md.
input bool   InpPropMode       = false;  // Prop challenge sizing on/off
input double InpPropScale      = 0.60;   // Fraction of the strategy to trade
input double InpPropInitial    = 0.0;    // Challenge starting balance, 0 = at attach
input double InpPropMaxLossPct = 10.0;   // Total loss limit, % of the INITIAL balance
input double InpPropCushionCap = 3.0;    // Cap on the cushion multiple
input bool   InpPropFlat       = false;  // Flat sizing instead of cushion sizing
input double InpDailyStopPct   = 0.0;    // Flatten for the day at this % loss, 0 = off

//--- Weekend ------------------------------------------------------------
//--- Between Friday's last quote and Sunday's first there is no fill at any
//--- price, so a breaker cannot act there. On this symbol that reopen carries
//--- two to three times the weeknight standard deviation and the single worst
//--- move in each split, while its mean is negative in both and significant in
//--- neither. Closing before it is what lets a funded account carry a size
//--- above 1.0 without a day that would have ended it.
input bool   InpFlatWeekend    = false;  // Close before the weekly shutdown
input int    InpWeekendFlatMin = 1000;   // Friday NY minute to be flat by (16:40)

//--- Plumbing -----------------------------------------------------------
input ulong  InpMagic          = 7710001;
input int    InpSlippagePts    = 20;
input bool   InpVerbose        = true;

//--- MQL5 compiles in one pass, so anything used before its body is defined
//--- needs a prototype here.
bool   ClosePartial(const double lots);
double HeldLots();
double PropMultiple();
bool   DailyBreaker();
bool   WeekendFlat();

CTrade   g_trade;
double   g_weight     = 0.0;      // what we believe we are holding
int      g_lastDayKey = -1;       // NY date of the last decision
double   g_peakEquity = 0.0;
bool     g_halted     = false;
int      g_serverGmtOffsetSec = 0;
double   g_propInitial = 0.0;    // challenge starting balance, latched at attach
double   g_effWeight   = 0.0;    // weight last actually set, cushion included
int      g_brokerDay   = -1;     // day index the equity baseline belongs to
double   g_dayStartEq  = 0.0;    // equity at the start of the broker day
int      g_dayStopKey  = -1;     // day the breaker fired, if any

#define SEC_PER_DAY 86400

//--- The band compares today's target against the weight the expert last set,
//--- not against a mark-to-market weight, so it has to survive a terminal
//--- restart or the first tick after one would force a needless rebalance.
//--- Keyed by magic so two instances never share state.
string WeightVarName() { return StringFormat("RML_%I64u_weight", InpMagic); }
string DayVarName()    { return StringFormat("RML_%I64u_daykey", InpMagic); }
string EffVarName()    { return StringFormat("RML_%I64u_effw",   InpMagic); }

void SaveState()
  {
   GlobalVariableSet(WeightVarName(), g_weight);
   GlobalVariableSet(EffVarName(), g_effWeight);
   GlobalVariableSet(DayVarName(), (double)g_lastDayKey);
  }

void LoadState()
  {
   if(GlobalVariableCheck(WeightVarName()))
      g_weight = GlobalVariableGet(WeightVarName());
   if(GlobalVariableCheck(EffVarName()))
      g_effWeight = GlobalVariableGet(EffVarName());
   if(GlobalVariableCheck(DayVarName()))
      g_lastDayKey = (int)GlobalVariableGet(DayVarName());
  }

//+------------------------------------------------------------------+
//| US daylight saving: second Sunday in March to first Sunday in    |
//| November. The instrument's halt moves with New York, not with    |
//| the server clock, and the broker has changed its UTC hours twice |
//| while never changing its New York hours - so New York is the     |
//| only clock worth keying off.                                     |
//+------------------------------------------------------------------+
datetime NthSunday(const int year, const int month, const int nth)
  {
   MqlDateTime t;
   t.year = year; t.mon = month; t.day = 1;
   t.hour = 0; t.min = 0; t.sec = 0;
   const datetime first = StructToTime(t);
   MqlDateTime f; TimeToStruct(first, f);
   const int offset = (7 - f.day_of_week) % 7;   // day_of_week: Sunday = 0
   return first + (datetime)((offset + 7 * (nth - 1)) * SEC_PER_DAY);
  }

bool IsUSDST(const datetime utc)
  {
   MqlDateTime t; TimeToStruct(utc, t);
   return utc >= NthSunday(t.year, 3, 2) && utc < NthSunday(t.year, 11, 1);
  }

datetime ServerToNY(const datetime server)
  {
   const datetime utc = server - (datetime)g_serverGmtOffsetSec;
   return utc - (datetime)((IsUSDST(utc) ? 4 : 5) * 3600);
  }

int NYMinuteOfDay(const datetime server)
  {
   MqlDateTime t; TimeToStruct(ServerToNY(server), t);
   return t.hour * 60 + t.min;
  }

int NYDayKey(const datetime server)
  {
   MqlDateTime t; TimeToStruct(ServerToNY(server), t);
   return t.year * 10000 + t.mon * 100 + t.day;
  }

//+------------------------------------------------------------------+
//| Build the series of US cash-session closes from M30 bars.        |
//|                                                                  |
//| 16:00 New York is an exact half-hour boundary, so the M30 bar    |
//| that *starts* at 15:30 New York is the last bar of the cash      |
//| session and its close is the session close. Working from M30     |
//| rather than M1 keeps the history request to a few thousand bars  |
//| instead of a few hundred thousand.                               |
//|                                                                  |
//| Bar zero is excluded unconditionally: it is still forming, and   |
//| a signal that reads a forming bar is a signal that reads the     |
//| future.                                                          |
//+------------------------------------------------------------------+
int BuildCashCloses(double &out[], const int wanted)
  {
   const int need = (int)MathMin(60000, (wanted + 20) * 48);
   MqlRates rates[];
   ArraySetAsSeries(rates, false);
   const int got = CopyRates(_Symbol, PERIOD_M30, 1, need, rates);
   if(got <= 0)
      return 0;

   const int targetMin = InpCashCloseMin - 30;    // bar start of the final bar
   double closes[];
   ArrayResize(closes, 0);
   for(int i = 0; i < got; i++)
     {
      if(NYMinuteOfDay(rates[i].time) != targetMin)
         continue;
      const int n = ArraySize(closes);
      ArrayResize(closes, n + 1);
      closes[n] = rates[i].close;
     }

   const int n = ArraySize(closes);
   const int take = (int)MathMin(n, wanted);
   ArrayResize(out, take);
   for(int i = 0; i < take; i++)
      out[i] = closes[n - take + i];        // ascending, most recent last
   return take;
  }

//+------------------------------------------------------------------+
//| The target weight, or -1 if there is not enough history yet.     |
//+------------------------------------------------------------------+
double TargetWeight(double &diagMa, double &diagVol, double &diagLast)
  {
   const int wanted = InpMaLen + InpVolLen + 5;
   double c[];
   const int n = BuildCashCloses(c, wanted);
   if(n < InpMaLen + InpVolLen + 1)
     {
      if(InpVerbose)
         PrintFormat("RML: only %d cash closes, need %d - standing aside",
                     n, InpMaLen + InpVolLen + 1);
      return -1.0;
     }

   double sum = 0.0;
   for(int i = n - InpMaLen; i < n; i++)
      sum += c[i];
   const double ma = sum / InpMaLen;

   double r[];
   ArrayResize(r, InpVolLen);
   double mean = 0.0;
   for(int i = 0; i < InpVolLen; i++)
     {
      const int j = n - InpVolLen + i;
      r[i] = MathLog(c[j] / c[j - 1]);
      mean += r[i];
     }
   mean /= InpVolLen;
   double var = 0.0;
   for(int i = 0; i < InpVolLen; i++)
      var += (r[i] - mean) * (r[i] - mean);
   // Sample standard deviation, matching the research code's rolling_std.
   const double sd = MathSqrt(var / (InpVolLen - 1));
   const double volAnn = sd * MathSqrt(252.0) * 100.0;

   diagMa = ma; diagVol = volAnn; diagLast = c[n - 1];
   if(volAnn <= 0.0)
      return 0.0;

   const double gate = (c[n - 1] > ma) ? 1.0 : InpGateFloor;
   double w = InpTargetVolPct / volAnn;
   if(w > InpMaxWeight) w = InpMaxWeight;
   if(w < 0.0)          w = 0.0;
   return w * gate;
  }

//+------------------------------------------------------------------+
//| Position helpers                                                 |
//+------------------------------------------------------------------+
double HeldLots()
  {
   double lots = 0.0;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      const ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
      if((ulong)PositionGetInteger(POSITION_MAGIC) != InpMagic) continue;
      const double v = PositionGetDouble(POSITION_VOLUME);
      lots += (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY) ? v : -v;
     }
   return lots;
  }

//+------------------------------------------------------------------+
//| The cushion multiple: how much room is left to the static floor,  |
//| as a fraction of the room the challenge started with.             |
//+------------------------------------------------------------------+
double PropMultiple()
  {
   if(!InpPropMode) return 1.0;
   //--- Flat sizing trades the same multiple every day. It is the right rule
   //--- when the objective is speed rather than survival: cushion sizing
   //--- shrinks the position as the account falls, which protects the total
   //--- floor by converting a fast failure into a slow one that never
   //--- resolves - and an attempt that never resolves is the most expensive
   //--- outcome of all, because it spends the calendar without spending the fee.
   if(InpPropFlat) return InpPropScale;
   const double initial = (InpPropInitial > 0.0) ? InpPropInitial : g_propInitial;
   if(initial <= 0.0) return 0.0;
   const double floorEq = initial * (1.0 - InpPropMaxLossPct / 100.0);
   const double room    = initial - floorEq;
   if(room <= 0.0) return 0.0;
   const double cushion = (AccountInfoDouble(ACCOUNT_EQUITY) - floorEq) / room;
   double m = InpPropScale * cushion;
   if(m < 0.0)                m = 0.0;
   if(m > InpPropScale * InpPropCushionCap) m = InpPropScale * InpPropCushionCap;
   return m;
  }

double SizingEquity()
  {
   if(InpEquityOverride > 0.0) return InpEquityOverride;
   return AccountInfoDouble(ACCOUNT_EQUITY);
  }

double LotsForWeight(const double weight, const double price)
  {
   const double contract = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_CONTRACT_SIZE);
   if(contract <= 0.0 || price <= 0.0) return 0.0;
   const double raw = weight * PropMultiple() * SizingEquity() / (price * contract);

   const double step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   const double minL = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   const double maxL = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   if(step <= 0.0) return 0.0;
   double lots = MathFloor(raw / step + 1e-9) * step;
   if(lots < minL) lots = (raw >= minL * 0.5) ? minL : 0.0;
   if(lots > maxL) lots = maxL;
   return NormalizeDouble(lots, 2);
  }

//+------------------------------------------------------------------+
//| Move the net position to `lots`, in one order.                   |
//+------------------------------------------------------------------+
bool Rebalance(const double targetLots)
  {
   const double held = HeldLots();
   double delta = targetLots - held;
   const double step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   if(MathAbs(delta) < step * 0.5)
      return true;

   g_trade.SetExpertMagicNumber(InpMagic);
   g_trade.SetDeviationInPoints(InpSlippagePts);

   // Closing before reversing keeps this correct on a hedging account, where
   // an opposing order would otherwise open a second position rather than net.
   if(held != 0.0 && ((held > 0 && targetLots < held) || (held < 0)))
     {
      const double reduce = MathMin(MathAbs(delta), MathAbs(held));
      if(!ClosePartial(reduce))
         return false;
      delta = targetLots - HeldLots();
      if(MathAbs(delta) < step * 0.5)
         return true;
     }

   if(delta > 0)
      return g_trade.Buy(NormalizeDouble(delta, 2), _Symbol, 0.0, 0.0, 0.0, "RML");
   return g_trade.Sell(NormalizeDouble(-delta, 2), _Symbol, 0.0, 0.0, 0.0, "RML");
  }

bool ClosePartial(const double lots)
  {
   double remaining = lots;
   const double step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   for(int i = PositionsTotal() - 1; i >= 0 && remaining > step * 0.5; i--)
     {
      const ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
      if((ulong)PositionGetInteger(POSITION_MAGIC) != InpMagic) continue;
      const double v = PositionGetDouble(POSITION_VOLUME);
      const double take = MathMin(v, remaining);
      g_trade.SetExpertMagicNumber(InpMagic);
      if(!g_trade.PositionClosePartial(ticket, NormalizeDouble(take, 2)))
         return false;
      remaining -= take;
     }
   return true;
  }

//+------------------------------------------------------------------+
//| Safety                                                           |
//+------------------------------------------------------------------+
bool SwapIsAcceptable()
  {
   if(InpSwapCheckBps <= 0.0) return true;
   const double swapLong = SymbolInfoDouble(_Symbol, SYMBOL_SWAP_LONG);
   const long   mode     = SymbolInfoInteger(_Symbol, SYMBOL_SWAP_MODE);
   const double price    = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double bps;

   if(mode == SYMBOL_SWAP_MODE_INTEREST_CURRENT ||
      mode == SYMBOL_SWAP_MODE_INTEREST_OPEN)
      bps = -swapLong / 365.0 * 100.0;                 // annual % -> bps/night
   else if(mode == SYMBOL_SWAP_MODE_POINTS)
      bps = -swapLong * _Point / price * 10000.0;
   else
      return true;   // a mode this expert cannot convert - let the human judge

   if(bps > InpSwapCheckBps)
     {
      PrintFormat("RML: swap is %.2f bps/night, above the %.2f limit - "
                  "not trading. Break-even is about 5 bps/night.",
                  bps, InpSwapCheckBps);
      return false;
     }
   return true;
  }

bool DrawdownKill()
  {
   const double eq = AccountInfoDouble(ACCOUNT_EQUITY);
   if(eq > g_peakEquity) g_peakEquity = eq;
   if(g_peakEquity <= 0.0) return false;
   const double dd = 100.0 * (1.0 - eq / g_peakEquity);
   if(!g_halted && dd >= InpKillDrawdownPct)
     {
      g_halted = true;
      PrintFormat("RML: KILL SWITCH - equity %.1f%% below peak. Flattening.", dd);
      ClosePartial(MathAbs(HeldLots()));
      g_weight = 0.0;
      g_effWeight = 0.0;
      SaveState();
     }
   return g_halted;
  }

//+------------------------------------------------------------------+
int OnInit()
  {
   g_serverGmtOffsetSec = (int)MathRound((double)(TimeCurrent() - TimeGMT()) / 3600.0) * 3600;
   g_peakEquity = AccountInfoDouble(ACCOUNT_EQUITY);
   //--- The challenge balance is latched ONCE. Re-latching it after a loss
   //--- would move the floor down with the account, which is exactly the
   //--- mistake the static rule punishes.
   g_propInitial = (InpPropInitial > 0.0)
                   ? InpPropInitial : AccountInfoDouble(ACCOUNT_BALANCE);
   g_weight = 0.0;
   LoadState();
   if(HeldLots() == 0.0)
     {
      g_weight = 0.0;      // no position, whatever the saved state claimed
      g_effWeight = 0.0;
     }

   if(InpMaLen < 20 || InpVolLen < 5)
     {
      Print("RML: MaLen >= 20 and VolLen >= 5, please.");
      return INIT_PARAMETERS_INCORRECT;
     }
   if(InpCashCloseMin % 30 != 0 || InpCashOpenMin % 30 != 0)
     {
      Print("RML: cash open and close must land on M30 boundaries.");
      return INIT_PARAMETERS_INCORRECT;
     }

   double ma = 0, vol = 0, last = 0;
   const double w = TargetWeight(ma, vol, last);
   PrintFormat("RML on %s | server-GMT %+d h | NY now %02d:%02d | US DST %s",
               _Symbol, g_serverGmtOffsetSec / 3600,
               NYMinuteOfDay(TimeCurrent()) / 60, NYMinuteOfDay(TimeCurrent()) % 60,
               IsUSDST(TimeGMT()) ? "on" : "off");
   if(w >= 0.0)
      PrintFormat("RML: close %.2f | SMA%d %.2f | vol %.1f%% | target weight %.2f",
                  last, InpMaLen, ma, vol, w);
   if(InpPropMode)
      PrintFormat("RML: PROP MODE. initial %.2f, floor %.2f, scale %.2f, "
                  "cushion multiple now %.3f",
                  g_propInitial, g_propInitial*(1.0 - InpPropMaxLossPct/100.0),
                  InpPropScale, PropMultiple());
   PrintFormat("RML: NO DIRECTIONAL EDGE IS CLAIMED. Swap is not in the "
               "backtest; break-even is about 5 bps/night.");
   return INIT_SUCCEEDED;
  }

//+------------------------------------------------------------------+
//| The daily circuit breaker.                                        |
//|                                                                   |
//| Cushion sizing protects the TOTAL floor and does nothing for the   |
//| daily one, because that resets each night however much room the    |
//| account has. At the sizes a short timeline needs, the daily floor  |
//| is where every failure comes from - so it needs its own rule.      |
//|                                                                   |
//| Flatten inside the limit and stand down until tomorrow. The loss   |
//| becomes known and survivable instead of being whatever the market  |
//| hands you.                                                        |
//+------------------------------------------------------------------+
bool DailyBreaker()
  {
   if(InpDailyStopPct <= 0.0) return false;

   const int today = (int)(TimeCurrent() / 86400);
   if(g_dayStartEq <= 0.0 || today != g_brokerDay)
     {
      g_brokerDay  = today;
      g_dayStartEq = AccountInfoDouble(ACCOUNT_EQUITY);
     }
   if(g_dayStopKey == today)
      return true;                     // already flat for the day

   const double eq = AccountInfoDouble(ACCOUNT_EQUITY);
   const double dd = 100.0 * (1.0 - eq / g_dayStartEq);
   if(dd >= InpDailyStopPct)
     {
      PrintFormat("RML: DAILY BREAKER at -%.2f%% (limit %.2f%%). Flat until tomorrow.",
                  dd, InpDailyStopPct);
      ClosePartial(MathAbs(HeldLots()));
      g_weight = 0.0;
      g_effWeight = 0.0;
      g_dayStopKey = today;
      SaveState();
      return true;
     }
   return false;
  }

//+------------------------------------------------------------------+
//| Flat across the weekly shutdown.                                  |
//|                                                                   |
//| The daily breaker is a promise to act; a gap is the market         |
//| refusing to let you. This is the only defence against the second,  |
//| and it is simply not being there. Holiday closures are not         |
//| detected - they are 4 of 205 multi-day shutdowns in dev and 3 of   |
//| 80 in validation, so the weekend rule covers 98% of the exposure.  |
//+------------------------------------------------------------------+
bool WeekendFlat()
  {
   if(!InpFlatWeekend) return false;

   MqlDateTime ny;
   TimeToStruct(ServerToNY(TimeCurrent()), ny);
   if(ny.day_of_week != 5 || NYMinuteOfDay(TimeCurrent()) < InpWeekendFlatMin)
      return false;

   const double held = MathAbs(HeldLots());
   if(held > 0.0)
     {
      PrintFormat("RML: flat for the weekend - closing %.2f lots", held);
      if(ClosePartial(held))
        {
         g_weight    = 0.0;
         g_effWeight = 0.0;
         SaveState();
        }
     }
   return true;                         // and no decisions until Monday
  }

//+------------------------------------------------------------------+
void OnTick()
  {
   if(DrawdownKill())
      return;
   if(WeekendFlat())
      return;
   if(DailyBreaker())
      return;

   const datetime now = TimeCurrent();
   const int minute = NYMinuteOfDay(now);
   const int dayKey = NYDayKey(now);

   if(dayKey == g_lastDayKey)
      return;
   if(minute < InpCashOpenMin || minute > InpCashOpenMin + InpDecisionWindow)
      return;

   const double spread = (SymbolInfoDouble(_Symbol, SYMBOL_ASK)
                          - SymbolInfoDouble(_Symbol, SYMBOL_BID)) / _Point;
   if(spread > InpMaxSpreadPts)
      return;                       // try again later in the decision window
   if(!SwapIsAcceptable())
     {
      g_lastDayKey = dayKey;
      return;
     }

   double ma = 0, vol = 0, last = 0;
   const double target = TargetWeight(ma, vol, last);
   if(target < 0.0)
     {
      g_lastDayKey = dayKey;
      return;
     }

   g_lastDayKey = dayKey;
   SaveState();

   //--- The band compares EFFECTIVE weights. In prop mode the cushion
   //--- multiple moves with equity even on days the strategy's own target
   //--- does not, and comparing raw targets would leave the position
   //--- stranded at yesterday's cushion - which is not what was simulated.
   const double propMult = PropMultiple();
   const double effTarget = target * propMult;
   if(MathAbs(effTarget - g_effWeight) <= InpBand)
     {
      if(InpVerbose)
         PrintFormat("RML: target %.2f x %.3f = %.2f, holding %.2f - inside the band",
                     target, propMult, effTarget, g_effWeight);
      g_weight = target;
      SaveState();
      return;
     }

   const double price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   const double lots  = LotsForWeight(target, price);
   PrintFormat("RML: rebalance | close %.2f SMA%d %.2f | vol %.1f%% | "
               "weight %.2f -> %.2f | prop x%.3f | lots %.2f -> %.2f",
               last, InpMaLen, ma, vol, g_weight, target,
               PropMultiple(), HeldLots(), lots);
   if(Rebalance(lots))
     {
      g_weight = target;
      g_effWeight = effTarget;
      SaveState();
     }
   else
      PrintFormat("RML: rebalance failed, retcode %d - weight left at %.2f",
                  g_trade.ResultRetcode(), g_weight);
  }

//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   PrintFormat("RML: stopping (reason %d). Position left as is: %.2f lots. "
               "This expert does not flatten on removal.", reason, HeldLots());
  }
//+------------------------------------------------------------------+

(** Core order and event types for the matching engine.

    Prices are integer [ticks] everywhere inside the engine. Integer comparison
    is exact, so a whole class of float-equality bugs in the matching path is
    unrepresentable; conversion to dollars happens only at the reporting
    boundary. Time priority within a price level is defined by an
    engine-assigned [entry_seq] rather than wall-clock time, so priority is
    total and unambiguous even for events sharing a timestamp. *)

type side =
  | Buy
  | Sell

(** [+1] for [Buy], [-1] for [Sell], so the crossing test can be written once
    for both sides instead of branching. *)
let sign = function Buy -> 1 | Sell -> -1

let opposite = function Buy -> Sell | Sell -> Buy
let side_to_string = function Buy -> "BUY" | Sell -> "SELL"

(** A live order resting in the book. [qty] is the [remaining] quantity and is
    mutated as fills occur; [entry_seq] fixes the order's place in its level's
    FIFO queue. *)
type resting_order = {
  order_id : int;
  owner : string;
  side : side;
  price : int;  (** ticks *)
  mutable qty : int;  (** remaining *)
  entry_seq : int;
}

(** A limit order, or the unfilled remainder of one, was added to the book. *)
type accepted = {
  a_time : float;
  a_order_id : int;
  a_owner : string;
  a_side : side;
  a_price : int;
  a_qty : int;  (** quantity actually rested: original minus immediate fills *)
}

(** One trade: an incoming (taker) order matched a resting (maker) order.

    The execution price is always the [maker]'s limit price, so a taker willing
    to trade through gets price improvement level by level. [f_pre_trade_mid] is
    the mid prevailing just before the incoming order began matching, recorded
    so that spread-capture accounting is measured against a mid this trade has
    not already moved. *)
type fill = {
  f_time : float;
  f_price : int;  (** ticks; the maker's resting price *)
  f_qty : int;
  f_taker_order_id : int;
  f_maker_order_id : int;
  f_taker_owner : string;
  f_maker_owner : string;
  f_taker_side : side;  (** the maker's side is the opposite by construction *)
  f_pre_trade_mid : float option;
}

(** An order was removed without (further) execution. [c_reason] is ["user"] for
    explicit cancels and ["unfilled_market"] for the remainder of a market order
    that exhausted available liquidity. *)
type cancelled = {
  c_time : float;
  c_order_id : int;
  c_owner : string;
  c_side : side;
  c_price : int option;  (** [None] for market remainders: they never rest *)
  c_qty : int;
  c_reason : string;
}

type event =
  | Accepted of accepted
  | Fill of fill
  | Cancelled of cancelled

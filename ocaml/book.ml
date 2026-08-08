(** A price-time-priority limit order book matching engine.

    Semantics are those of a standard continuous double auction:

    - {b Price priority}: an incoming order trades against the best-priced
      opposite orders first (highest bid / lowest ask).
    - {b Time priority (FIFO)}: within a price level, orders execute in arrival
      order, defined by an engine-assigned sequence number.
    - {b Execution price}: every fill prints at the {i resting} order's limit
      price, so an aggressive order that crosses several levels gets price
      improvement level by level ("walking the book").
    - {b Marketable limit orders} match immediately for whatever is available
      inside their limit, then rest the remainder.
    - {b Market orders} never rest; any unmatched remainder is discarded and
      reported as [Cancelled] with reason ["unfilled_market"].

    {2 Data structures}

    Each side is a map from price tick to that level's FIFO queue, held as a
    list with the oldest order at the head. The Python implementation of this
    engine pairs a hash map of levels with a lazily-cleaned binary heap of
    prices; here a balanced-tree [Map] subsumes both, since [max_binding] and
    [min_binding] give the best bid and ask directly in [O(log n)]. Empty levels
    are deleted eagerly, so every price in the map is live and no stale-entry
    cleanup is needed.

    Cancellation removes the order from its level immediately rather than
    tombstoning it. That is [O(level size)], but it buys a much stronger
    invariant, which is that every order reachable from a queue is live, and
    level sizes in this simulation are small. *)

open Orders

module IntMap = Map.Make (Int)

type t = {
  mutable bids : resting_order list IntMap.t;
  mutable asks : resting_order list IntMap.t;
  mutable orders : resting_order IntMap.t;  (** live orders by id, for cancel *)
  mutable next_order_id : int;
  mutable next_entry_seq : int;
}

let create () =
  {
    bids = IntMap.empty;
    asks = IntMap.empty;
    orders = IntMap.empty;
    next_order_id = 1;
    next_entry_seq = 1;
  }

let levels book = function Buy -> book.bids | Sell -> book.asks

let set_levels book side m =
  match side with Buy -> book.bids <- m | Sell -> book.asks <- m

(* ------------------------------------------------------------------ views *)

let best_bid book =
  match IntMap.max_binding_opt book.bids with Some (p, _) -> Some p | None -> None

let best_ask book =
  match IntMap.min_binding_opt book.asks with Some (p, _) -> Some p | None -> None

let best_price book = function Buy -> best_bid book | Sell -> best_ask book

(** Mid price in ticks; may be a half-integer, hence [float]. [None] if either
    side is empty. *)
let mid book =
  match (best_bid book, best_ask book) with
  | Some b, Some a -> Some (float_of_int (b + a) /. 2.)
  | _ -> None

let spread book =
  match (best_bid book, best_ask book) with
  | Some b, Some a -> Some (a - b)
  | _ -> None

let level_qty q = List.fold_left (fun acc o -> acc + o.qty) 0 q

(** Aggregate [(price, total_qty)] per level, best first. *)
let depth book side n_levels =
  let bindings = IntMap.bindings (levels book side) in
  (* [IntMap.bindings] is ascending by price: best is the last for bids, the
     first for asks. *)
  let ordered = match side with Buy -> List.rev bindings | Sell -> bindings in
  let ordered =
    match n_levels with
    | None -> ordered
    | Some k -> List.filteri (fun i _ -> i < k) ordered
  in
  List.map (fun (p, q) -> (p, level_qty q)) ordered

let get_order book order_id = IntMap.find_opt order_id book.orders
let order_count book = IntMap.cardinal book.orders

(* --------------------------------------------------------------- matching *)

let new_order_id book =
  let id = book.next_order_id in
  book.next_order_id <- id + 1;
  id

let rest_order book ~order_id ~owner ~side ~price ~qty =
  let order =
    { order_id; owner; side; price; qty; entry_seq = book.next_entry_seq }
  in
  book.next_entry_seq <- book.next_entry_seq + 1;
  let m = levels book side in
  let q = Option.value (IntMap.find_opt price m) ~default:[] in
  (* Append to the tail: newest order has lowest time priority. *)
  set_levels book side (IntMap.add price (q @ [ order ]) m);
  book.orders <- IntMap.add order_id order book.orders

(** Match an incoming order against the opposite side. Mutates the book and
    returns [(remaining, events)] with events in chronological order.

    [limit_price] is [None] for market orders, which cross at any price. *)
let match_incoming book ~taker_id ~taker_owner ~side ~qty ~limit_price ~time =
  let pre_trade_mid = mid book in
  let opp = opposite side in
  let remaining = ref qty in
  let events = ref [] in
  let keep_going = ref true in
  while !keep_going && !remaining > 0 do
    match best_price book opp with
    | None -> keep_going := false
    | Some best ->
        (* Crossing test, written sign-uniformly: a Buy crosses when its limit
           is at or above the best ask, a Sell when at or below the best bid. *)
        let crosses =
          match limit_price with
          | None -> true
          | Some lp -> sign side * (lp - best) >= 0
        in
        if not crosses then keep_going := false
        else begin
          let queue = IntMap.find best (levels book opp) in
          (* Walk this level's FIFO, consuming makers until the incoming order
             is filled or the level is exhausted. Returns what is left of the
             queue plus the fills generated, newest first. *)
          let rec walk q acc =
            if !remaining <= 0 then (q, acc)
            else
              match q with
              | [] -> ([], acc)
              | maker :: rest ->
                  let traded = min !remaining maker.qty in
                  maker.qty <- maker.qty - traded;
                  remaining := !remaining - traded;
                  let ev =
                    Fill
                      {
                        f_time = time;
                        f_price = maker.price;
                        f_qty = traded;
                        f_taker_order_id = taker_id;
                        f_maker_order_id = maker.order_id;
                        f_taker_owner = taker_owner;
                        f_maker_owner = maker.owner;
                        f_taker_side = side;
                        f_pre_trade_mid = pre_trade_mid;
                      }
                  in
                  if maker.qty = 0 then begin
                    book.orders <- IntMap.remove maker.order_id book.orders;
                    walk rest (ev :: acc)
                  end
                  else (maker :: rest, ev :: acc)
          in
          let leftover, fills = walk queue [] in
          events := fills @ !events;
          match leftover with
          | [] -> set_levels book opp (IntMap.remove best (levels book opp))
          | _ -> set_levels book opp (IntMap.add best leftover (levels book opp))
        end
  done;
  (!remaining, List.rev !events)

(* ---------------------------------------------------------------- actions *)

let check_qty qty = if qty <= 0 then invalid_arg "qty must be a positive integer"

(** Submit a limit order. It first matches against any crossing liquidity as the
    taker, then rests the remainder at [price]. Returns [(order_id, events)]. *)
let submit_limit book ~owner ~side ~price ~qty ~time =
  check_qty qty;
  let order_id = new_order_id book in
  let remaining, events =
    match_incoming book ~taker_id:order_id ~taker_owner:owner ~side ~qty
      ~limit_price:(Some price) ~time
  in
  if remaining > 0 then begin
    rest_order book ~order_id ~owner ~side ~price ~qty:remaining;
    let accepted =
      Accepted
        {
          a_time = time;
          a_order_id = order_id;
          a_owner = owner;
          a_side = side;
          a_price = price;
          a_qty = remaining;
        }
    in
    (order_id, events @ [ accepted ])
  end
  else (order_id, events)

(** Submit a market order. Quantity that cannot be matched is discarded and
    reported as [Cancelled] with reason ["unfilled_market"]. *)
let submit_market book ~owner ~side ~qty ~time =
  check_qty qty;
  let order_id = new_order_id book in
  let remaining, events =
    match_incoming book ~taker_id:order_id ~taker_owner:owner ~side ~qty
      ~limit_price:None ~time
  in
  if remaining > 0 then
    let unfilled =
      Cancelled
        {
          c_time = time;
          c_order_id = order_id;
          c_owner = owner;
          c_side = side;
          c_price = None;
          c_qty = remaining;
          c_reason = "unfilled_market";
        }
    in
    (order_id, events @ [ unfilled ])
  else (order_id, events)

(** Cancel a resting order. Returns [None] if the order is unknown or already
    fully filled, mirroring real venues where cancels race with fills. *)
let cancel book ~order_id ~time =
  match IntMap.find_opt order_id book.orders with
  | None -> None
  | Some order ->
      book.orders <- IntMap.remove order_id book.orders;
      let m = levels book order.side in
      let q = IntMap.find order.price m in
      let q' = List.filter (fun o -> o.order_id <> order_id) q in
      (match q' with
      | [] -> set_levels book order.side (IntMap.remove order.price m)
      | _ -> set_levels book order.side (IntMap.add order.price q' m));
      Some
        {
          c_time = time;
          c_order_id = order_id;
          c_owner = order.owner;
          c_side = order.side;
          c_price = Some order.price;
          c_qty = order.qty;
          c_reason = "user";
        }

(* ------------------------------------------------------------- invariants *)

(** Assert internal consistency. Used by tests after every operation; never
    called on the hot path. *)
let check_invariants book =
  (match (best_bid book, best_ask book) with
  | Some b, Some a ->
      if b >= a then
        failwith (Printf.sprintf "book is crossed: bid %d >= ask %d" b a)
  | _ -> ());
  let seen = ref [] in
  List.iter
    (fun side ->
      IntMap.iter
        (fun price q ->
          if q = [] then failwith (Printf.sprintf "empty level %d left in book" price);
          let seqs = List.map (fun o -> o.entry_seq) q in
          if seqs <> List.sort compare seqs then
            failwith (Printf.sprintf "FIFO order violated at level %d" price);
          List.iter
            (fun o ->
              if o.qty <= 0 then
                failwith (Printf.sprintf "zero-qty order %d in book" o.order_id);
              if o.price <> price || o.side <> side then
                failwith (Printf.sprintf "order %d filed at wrong level" o.order_id);
              (match IntMap.find_opt o.order_id book.orders with
              | Some o' when o' == o -> ()
              | _ ->
                  failwith
                    (Printf.sprintf "order %d missing from index" o.order_id));
              seen := o.order_id :: !seen)
            q)
        (levels book side))
    [ Buy; Sell ];
  let indexed = List.map fst (IntMap.bindings book.orders) in
  if List.sort compare !seen <> List.sort compare indexed then
    failwith "order index out of sync with levels"

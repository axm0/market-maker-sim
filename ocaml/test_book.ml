(** Tests for the OCaml matching engine.

    These mirror the invariants the Python engine is tested against: price
    priority, FIFO time priority within a level, execution at the maker's price
    (price improvement when walking the book), correct resting of remainders,
    and that market orders never rest. [Book.check_invariants] runs after every
    mutation, so a structural violation fails the test that caused it. *)

open Orders

let failures = ref 0
let checks = ref 0

let check name cond =
  incr checks;
  if not cond then begin
    incr failures;
    Printf.printf "  FAIL  %s\n" name
  end

let check_int name ~expected ~actual =
  incr checks;
  if expected <> actual then begin
    incr failures;
    Printf.printf "  FAIL  %s (expected %d, got %d)\n" name expected actual
  end

let fills events =
  List.filter_map (function Fill f -> Some f | _ -> None) events

let accepteds events =
  List.filter_map (function Accepted a -> Some a | _ -> None) events

let cancels events =
  List.filter_map (function Cancelled c -> Some c | _ -> None) events

(* --------------------------------------------------------------- fixtures *)

(** A book with asks resting at 101, 102 and bids at 99, 98. *)
let two_sided () =
  let b = Book.create () in
  ignore (Book.submit_limit b ~owner:"mm" ~side:Sell ~price:101 ~qty:10 ~time:0.);
  ignore (Book.submit_limit b ~owner:"mm" ~side:Sell ~price:102 ~qty:10 ~time:0.);
  ignore (Book.submit_limit b ~owner:"mm" ~side:Buy ~price:99 ~qty:10 ~time:0.);
  ignore (Book.submit_limit b ~owner:"mm" ~side:Buy ~price:98 ~qty:10 ~time:0.);
  Book.check_invariants b;
  b

(* ------------------------------------------------------------------ tests *)

let test_best_prices_and_mid () =
  let b = two_sided () in
  check "best bid is 99" (Book.best_bid b = Some 99);
  check "best ask is 101" (Book.best_ask b = Some 101);
  check "mid is 100" (Book.mid b = Some 100.);
  check "spread is 2" (Book.spread b = Some 2)

let test_price_priority () =
  (* A buy for 10 must hit the 101 ask, not the 102 one. *)
  let b = two_sided () in
  let _, events = Book.submit_market b ~owner:"taker" ~side:Buy ~qty:10 ~time:1. in
  Book.check_invariants b;
  let fs = fills events in
  check_int "one fill" ~expected:1 ~actual:(List.length fs);
  check_int "filled at the best ask" ~expected:101 ~actual:(List.hd fs).f_price;
  check "101 level is gone" (Book.best_ask b = Some 102)

let test_time_priority_is_fifo () =
  (* Two makers at the same price: the earlier one must fill first. *)
  let b = Book.create () in
  let first, _ = Book.submit_limit b ~owner:"early" ~side:Sell ~price:101 ~qty:5 ~time:0. in
  let second, _ = Book.submit_limit b ~owner:"late" ~side:Sell ~price:101 ~qty:5 ~time:0. in
  let _, events = Book.submit_market b ~owner:"taker" ~side:Buy ~qty:5 ~time:1. in
  Book.check_invariants b;
  let fs = fills events in
  check_int "one fill" ~expected:1 ~actual:(List.length fs);
  check_int "earlier order filled" ~expected:first ~actual:(List.hd fs).f_maker_order_id;
  check "later order still resting" (Book.get_order b second <> None)

let test_walks_the_book_with_price_improvement () =
  (* A large marketable buy consumes 101 then 102, printing each at the
     maker's price rather than at a single blended price. *)
  let b = two_sided () in
  let _, events = Book.submit_limit b ~owner:"taker" ~side:Buy ~price:102 ~qty:15 ~time:1. in
  Book.check_invariants b;
  let fs = fills events in
  check_int "two fills" ~expected:2 ~actual:(List.length fs);
  check_int "first prints at 101" ~expected:101 ~actual:(List.nth fs 0).f_price;
  check_int "first fills 10" ~expected:10 ~actual:(List.nth fs 0).f_qty;
  check_int "second prints at 102" ~expected:102 ~actual:(List.nth fs 1).f_price;
  check_int "second fills 5" ~expected:5 ~actual:(List.nth fs 1).f_qty;
  check "nothing rested: order fully filled" (accepteds events = [])

let test_marketable_limit_rests_remainder () =
  (* Buy 25 at 101 with only 10 available there: 10 fill, 15 rest as the bid. *)
  let b = two_sided () in
  let _, events = Book.submit_limit b ~owner:"taker" ~side:Buy ~price:101 ~qty:25 ~time:1. in
  Book.check_invariants b;
  check_int "one fill" ~expected:1 ~actual:(List.length (fills events));
  let a = List.hd (accepteds events) in
  check_int "remainder rested" ~expected:15 ~actual:a.a_qty;
  check "remainder is the new best bid" (Book.best_bid b = Some 101)

let test_market_order_never_rests () =
  (* Demand 50 against 20 of resting asks: 20 fill, 30 are discarded. *)
  let b = two_sided () in
  let _, events = Book.submit_market b ~owner:"taker" ~side:Buy ~qty:50 ~time:1. in
  Book.check_invariants b;
  check_int "two fills" ~expected:2 ~actual:(List.length (fills events));
  let c = List.hd (cancels events) in
  check_int "remainder discarded" ~expected:30 ~actual:c.c_qty;
  check "reported as unfilled_market" (c.c_reason = "unfilled_market");
  check "ask side is empty" (Book.best_ask b = None);
  check "taker did not rest a bid" (Book.best_bid b = Some 99)

let test_partial_fill_keeps_priority () =
  (* A maker partially filled stays at the head of its level with the
     remaining quantity. *)
  let b = Book.create () in
  let maker, _ = Book.submit_limit b ~owner:"mm" ~side:Sell ~price:101 ~qty:10 ~time:0. in
  ignore (Book.submit_market b ~owner:"taker" ~side:Buy ~qty:4 ~time:1.);
  Book.check_invariants b;
  match Book.get_order b maker with
  | None -> check "maker still resting" false
  | Some o -> check_int "remaining quantity" ~expected:6 ~actual:o.qty

let test_cancel () =
  let b = two_sided () in
  let id, _ = Book.submit_limit b ~owner:"mm" ~side:Buy ~price:100 ~qty:7 ~time:0. in
  check "new order is the best bid" (Book.best_bid b = Some 100);
  (match Book.cancel b ~order_id:id ~time:1. with
  | None -> check "cancel returned an event" false
  | Some c ->
      check_int "cancelled quantity" ~expected:7 ~actual:c.c_qty;
      check "reason is user" (c.c_reason = "user"));
  Book.check_invariants b;
  check "best bid reverts to 99" (Book.best_bid b = Some 99);
  check "second cancel is a no-op" (Book.cancel b ~order_id:id ~time:2. = None)

let test_non_crossing_limit_just_rests () =
  let b = two_sided () in
  let _, events = Book.submit_limit b ~owner:"mm" ~side:Buy ~price:100 ~qty:5 ~time:1. in
  Book.check_invariants b;
  check "no fills" (fills events = []);
  check_int "one acceptance" ~expected:1 ~actual:(List.length (accepteds events));
  check "book is not crossed" (Book.best_bid b = Some 100 && Book.best_ask b = Some 101)

let test_depth_is_aggregated_best_first () =
  let b = two_sided () in
  ignore (Book.submit_limit b ~owner:"mm2" ~side:Sell ~price:101 ~qty:3 ~time:0.);
  Book.check_invariants b;
  match Book.depth b Sell None with
  | (p0, q0) :: (p1, _) :: _ ->
      check_int "best ask level first" ~expected:101 ~actual:p0;
      check_int "quantities aggregated" ~expected:13 ~actual:q0;
      check_int "next level after" ~expected:102 ~actual:p1
  | _ -> check "two ask levels present" false

let test_pre_trade_mid_is_recorded_before_the_trade () =
  (* The mid recorded on a fill must be the one prevailing before the incoming
     order moved the book, which is what spread-capture accounting needs. *)
  let b = two_sided () in
  let _, events = Book.submit_market b ~owner:"taker" ~side:Buy ~qty:10 ~time:1. in
  let f = List.hd (fills events) in
  check "pre-trade mid is 100" (f.f_pre_trade_mid = Some 100.)

let test_rejects_non_positive_quantity () =
  let b = Book.create () in
  let raised =
    try
      ignore (Book.submit_limit b ~owner:"mm" ~side:Buy ~price:100 ~qty:0 ~time:0.);
      false
    with Invalid_argument _ -> true
  in
  check "zero quantity is rejected" raised

(* ------------------------------------------------------------------- main *)

let () =
  let tests =
    [
      ("best prices and mid", test_best_prices_and_mid);
      ("price priority", test_price_priority);
      ("time priority is FIFO", test_time_priority_is_fifo);
      ("walks the book with price improvement", test_walks_the_book_with_price_improvement);
      ("marketable limit rests remainder", test_marketable_limit_rests_remainder);
      ("market order never rests", test_market_order_never_rests);
      ("partial fill keeps priority", test_partial_fill_keeps_priority);
      ("cancel", test_cancel);
      ("non-crossing limit just rests", test_non_crossing_limit_just_rests);
      ("depth is aggregated, best first", test_depth_is_aggregated_best_first);
      ("pre-trade mid recorded before trade", test_pre_trade_mid_is_recorded_before_the_trade);
      ("rejects non-positive quantity", test_rejects_non_positive_quantity);
    ]
  in
  List.iter
    (fun (name, f) ->
      Printf.printf "%-42s" name;
      let before = !failures in
      f ();
      if !failures = before then print_string "ok\n" else print_newline ())
    tests;
  Printf.printf "\n%d checks, %d failures\n" !checks !failures;
  if !failures > 0 then exit 1

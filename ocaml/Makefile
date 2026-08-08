# Build with the stock OCaml compiler: no opam switch or dune required.
# Module order matters: Orders defines the types Book depends on.

SOURCES = orders.ml book.ml
TEST_SOURCES = $(SOURCES) test_book.ml

.PHONY: test clean

test: test_book
	./test_book

test_book: $(TEST_SOURCES)
	ocamlc -o test_book $(TEST_SOURCES)

clean:
	rm -f test_book *.cmi *.cmo *.cmx *.o

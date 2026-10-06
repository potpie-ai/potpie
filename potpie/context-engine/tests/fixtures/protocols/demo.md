# Synthetic Demo contract (test data, not a real protocol)

Demo revision 1, profile Test: Query request (client-to-server, discriminator Q)
contains header.Status at ordinal 0, uint8 at byte offset 0 from PDU origin.
Allowed raw values: integer 0 IDLE, integer 2 BUSY, string "2" TEXT, false OFF.
The nested payload.status field is optional when header.Status == 2, offset unknown.
Two response definitions Answer and Failure correspond to Query. Event Tick is
one-way. Query discriminator q is a different, case-sensitive message.
Revision 2 adds a units field; it does not overwrite revision 1.
DemoService can send Query and receive both responses; DemoPeer reverses these
capabilities. Synthetic src/decoder.py implements Query. These are declarations,
not evidence that packets were sent. The fixture also generates a 300-field
layout, missing evidence, source-scoped unknown revision, and conflicting extract.

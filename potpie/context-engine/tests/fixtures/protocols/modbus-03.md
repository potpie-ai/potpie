# Modbus function 03 normal PDU layouts

Fixture facts derived from Modbus Application Protocol Specification V1.1b3,
section 6.3, page 15 (2012-04-26):
https://www.modbus.org/file/secure/modbusprotocolspecification.pdf

The request reads holding registers. Its one-byte function discriminator is 3,
followed by a two-byte starting address (0 through 65535) and a two-byte register
quantity (1 through 125). Multi-byte address and register values use big endian
order. Offsets from the PDU start are 0, 1 and 3 bytes respectively.

The normal response starts with discriminator 3, followed by a one-byte byte
count, followed by register values. The byte count is twice the requested
quantity; each register occupies two bytes. Offsets are 0, 1 and 2 bytes.

This fixture materializes the normal request and response only. Exception
responses, transport headers, runtime exchanges and device implementations are
outside its coverage. Service and CodeAsset links elsewhere in the fixture are
synthetic and do not claim real Modbus participants or implementations.

pragma solidity ^0.8.27;
contract Unit {
    bool public completed;
    function execute(address payable receiver) external payable {
        completed = true; (bool ok,) = receiver.call{value: msg.value}(""); require(ok);
    }
    
}
